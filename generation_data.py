import asyncio
import json
import time
from utils.llm_config import llm_gemini, llm_o1, llm_4o_idea, llm_claude_s2
from utils.pre_process_prompt import PRE_PROCESS_PROMPT

async def generation_data(system_prompt, user_prompt, llm):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await llm(messages=messages, max_tokens=4096, stream=False, temperature=0)
        text = response.choices[0].message.content
        text_to_json = json.loads(text)
    except Exception as e:
        print("报错如下：" ,e)
        return False
    
    # 读取现有数据
    try:
        with open("./dataset/data.json", "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    except json.JSONDecodeError:
        existing_data = []
    
    # 将新数据追加到现有数据中
    existing_data.append(text_to_json)
    
    # 写回文件
    with open("./dataset/data.json", "w") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    return True

def data_pre_process():
    with open("./dataset/origin_data/ast_result.json", "r") as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    data = data_pre_process()
    for item in data:
        json_str = json.dumps(item, ensure_ascii=False, indent=2)
        flag= asyncio.run(generation_data(PRE_PROCESS_PROMPT, json_str, llm_claude_s2))
        if not flag:
            repeat_times = 0
            while repeat_times < 5:
                time.sleep(repeat_times+1)
                flag= asyncio.run(generation_data(PRE_PROCESS_PROMPT, json_str, llm_o1))
                if flag:
                    break
                else:
                    repeat_times+=1


