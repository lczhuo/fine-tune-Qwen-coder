from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

# 检查可用设备（优先使用MPS加速）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集预处理 ----------------------------------------------------------------
# 建议将原始数据预处理为以下格式的CSV/JSON：
# {"instruction": "...", "input": "...", "output": "..."}

# 加载并预处理数据集
def load_dataset(path):
    df = pd.read_json(path)
    return Dataset.from_pandas(df)

ds = load_dataset('./dataset/input.json')

# 初始化分词器 ---------------------------------------------------------------
model_path = './Qwen/Qwen2.5-Coder-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True,
    pad_token="<|endoftext|>"  # 显式设置pad_token
)

# 数据处理函数 ---------------------------------------------------------------
def process_func(example):
    # 严格遵循官方对话模板格式
    system_msg = "你是一个专业的代码助手，能够根据需求生成高质量的代码。"
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ],
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 统一编码（使用truncation代替手动截断）
    tokenized = tokenizer(
        prompt,
        max_length=4096,
        truncation=True,
        padding=False  # 后续由DataCollator处理
    )
    
    # 对齐labels（仅预测assistant部分）
    # 查找第一个<|im_start|>assistant的位置
    sep_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
    try:
        assistant_pos = tokenized["input_ids"].index(sep_token_id) + 1
    except ValueError:
        assistant_pos = 0
    
    labels = [-100] * assistant_pos + tokenized["input_ids"][assistant_pos:]
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# 处理数据集（启用内存映射加速）
tokenized_ds = ds.map(
    process_func,
    remove_columns=ds.column_names,
    load_from_cache_file=True,
    desc="Processing dataset"
)

# 划分训练验证集（9:1）
split_ds = tokenized_ds.train_test_split(test_size=0.1, seed=42)

# 模型配置 ---------------------------------------------------------------
# 加载基础模型（适配Apple Silicon）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # MPS建议使用bfloat16或float32
    device_map={"": device},      # 显式指定设备
    trust_remote_code=True,
    low_cpu_mem_usage=True      # 启用低内存模式
)
model.gradient_checkpointing_enable()  # 内存优化

# LoRA配置（经测试适配Qwen架构）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=4,                # 提升秩以获得更好效果
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数

# 训练配置 ---------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./output",
    optim="adamw_torch",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # 序列长度调整
    max_grad_norm=1.0,                   # 防止梯度爆炸
    
    # 批处理配置（适配18GB内存）
    per_device_train_batch_size=1,      # MPS可以适当增大
    gradient_accumulation_steps=10,      # 总batch_size=16
    num_train_epochs=3,
    
    # 训练监控
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=1,
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 设备优化
    torch_compile=False,                 # 禁用编译优化
    fp16=False,                  # MPS暂不支持混合精度
    bf16=torch.cuda.is_bf16_supported,  # Apple芯片无需开启
    gradient_checkpointing=True,
    
    # 禁用不兼容功能
    deepspeed=None               # MPS不支持DeepSpeed
)

# 使用官方数据整理器
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8  # 对齐显存访问
)

# 自定义评估指标
def compute_metrics(eval_pred):
    """
    综合评估指标计算函数，包含：
    - BLEU-4 (句子级平滑处理)
    - ROUGE-L (F1分数)
    - 单词准确率 (Exact Match)
    - 字符级编辑相似度
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)  # 转换logits为token ids

    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 处理标签中的-100（忽略padding位置）
    decoded_labels = []
    for label in labels:
        label_ids = [token_id for token_id in label if token_id != -100]
        decoded_labels.append(tokenizer.decode(label_ids, skip_special_tokens=True))

    # 初始化指标计算器
    rouge_calc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4  # BLEU平滑函数

    # 初始化统计量
    bleu_scores = []
    rouge_scores = []
    exact_matches = 0
    edit_similarities = []

    for pred, ref in zip(decoded_preds, decoded_labels):
        ########################
        # 1. BLEU-4 计算（句子级）
        ########################
        # 将文本转换为token列表（BLEU需要tokenized输入）
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]  # 注意：BLEU需要多个参考译文列表
        
        # 计算句子BLEU（使用平滑处理避免0分）
        bleu = corpus_bleu(
            [ref_tokens], 
            [pred_tokens],
            smoothing_function=smoothie,
            weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4权重
        )
        bleu_scores.append(bleu)

        ########################
        # 2. ROUGE-L 计算
        ########################
        rouge = rouge_calc.score(ref, pred)
        rouge_scores.append(rouge['rougeL'].fmeasure)

        ########################
        # 3. 完全匹配率
        ########################
        exact_matches += 1 if pred.strip() == ref.strip() else 0

        ########################
        # 4. 编辑相似度
        ########################
        # 计算Levenshtein距离
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        distance = edit_distance(pred, ref)
        max_len = max(len(pred), len(ref))
        similarity = 1 - distance / max_len if max_len > 0 else 1.0
        edit_similarities.append(similarity)

    # 汇总指标
    return {
        "bleu-4": np.mean(bleu_scores),              # 数据集级BLEU
        "rouge-L": np.mean(rouge_scores),            # 平均ROUGE-L F1
        "exact_match": exact_matches / len(labels),  # 完全匹配率
        "edit_sim": np.mean(edit_similarities),      # 平均编辑相似度
        "combined_score": (np.mean(bleu_scores) + np.mean(rouge_scores)) / 2
    }

# 在Trainer中添加内存优化回调
from transformers import TrainerCallback

class MemoryOptimizerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.mps.empty_cache()  # 每步清空缓存
# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_ds["train"],
    eval_dataset=split_ds["test"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[MemoryOptimizerCallback()],
)

# 开始训练
trainer.train()

# 模型保存 ---------------------------------------------------------------
# 保存完整适配器
model.save_pretrained("./output/final_adapter")

# 推理测试 ---------------------------------------------------------------
def generate_response(prompt, use_lora=True):
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    ).eval()
    
    # 加载LoRA适配器
    if use_lora:
        model = PeftModel.from_pretrained(
            base_model,
            "./output/final_adapter",
            device_map={"": device}
        )
    else:
        model = base_model

    # 构建对话模板
    messages = [
        {"role": "system", "content": "你是一个专业的代码助手"},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 生成配置
    gen_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id
    }

    # 生成回复
    outputs = model.generate(inputs, **gen_config)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

# 测试示例
test_prompt = "用Python实现快速排序"
print("原始模型:", generate_response(test_prompt, use_lora=False))
print("微调模型:", generate_response(test_prompt, use_lora=True))