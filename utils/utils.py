import json
import os
import tiktoken

def format_data_input():
    try:
        # 检查目录是否存在
        if not os.path.exists("./dataset"):
            os.makedirs("./dataset")
            print("创建dataset目录")

        # 检查输入文件是否存在
        if not os.path.exists("./dataset/data.json"):
            print("输入文件 data.json 不存在")
            return

        with open("./dataset/data.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据是否为空或非列表类型
        if not data or not isinstance(data, list):
            print("数据为空或格式不正确")
            return

        formatted_items = []

        for index, item in enumerate(data):
            try:
                # 检查必要字段
                if not all(key in item for key in ["code_explanation", "code_content"]):
                    print(f"跳过第 {index} 条数据：缺少必要字段")
                    continue

                if "summary" not in item["code_explanation"]:
                    print(f"跳过第 {index} 条数据：缺少 summary")
                    continue

                code_content = item["code_content"]
                if not isinstance(code_content, dict):
                    print(f"跳过第 {index} 条数据：code_content 格式不正确")
                    continue

                # 检查代码内容
                cleaned_code = code_content.get("cleaned_code", "")
                raw_code = code_content.get("raw_code", "")

                if not cleaned_code and not raw_code:
                    print(f"跳过第 {index} 条数据：没有有效的代码内容")
                    continue

                # 构建新项目
                summary = item["code_explanation"]["summary"]
                detailed_description = item["code_explanation"].get("detailed_description", "")
                
                instruction = f"{summary}。{detailed_description}" if detailed_description else summary
                output = cleaned_code if cleaned_code else raw_code

                new_item = {
                    "instruction": instruction.strip(),
                    "output": output.strip()
                }
                
                formatted_items.append(new_item)

            except AttributeError as e:
                print(f"处理第 {index} 条数据时出现属性错误: {e}")
                continue
            except Exception as e:
                print(f"处理第 {index} 条数据时出现未知错误: {e}")
                continue

        # 检查处理后的数据是否为空
        if not formatted_items:
            print("没有有效的数据被处理")
            return

        try:
            with open("./dataset/input.json", "w", encoding='utf-8') as f:
                json.dump(formatted_items, f, ensure_ascii=False, indent=2)
            print(f"成功处理 {len(formatted_items)} 条数据")
        
        except IOError as e:
            print(f"写入文件时出错: {e}")

    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确")
    except json.JSONDecodeError as e:
        print(f"JSON解码错误，请检查文件格式: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

def count_tokens(path = "./dataset/input.json"):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)

    encoding = tiktoken.encoding_for_model("gpt-4o")
    max_instruction_tokens = 0
    max_output_tokens = 0
    sum_instruction_tokens = 0
    sum_output_tokens = 0
    for item in data:
        instruction = item["instruction"]
        output = item["output"]
        instruction_tokens = len(encoding.encode(instruction))
        output_tokens = len(encoding.encode(output))
        max_instruction_tokens = max(max_instruction_tokens, instruction_tokens)
        max_output_tokens = max(max_output_tokens, output_tokens)
        sum_instruction_tokens += instruction_tokens
        sum_output_tokens += output_tokens
    avg_instruction_tokens = round(sum_instruction_tokens / len(data), 2)
    avg_output_tokens = round(sum_output_tokens / len(data), 2)

    return {
        "max_instruction_tokens": max_instruction_tokens,
        "max_output_tokens": max_output_tokens,
        "avg_instruction_tokens": avg_instruction_tokens,
        'avg_output_tokens': avg_output_tokens,
    }

if __name__ == "__main__":
    # format_data_input()
    result = count_tokens()
    for key, value in result.items():
        print(f"{key}: {value}")
    """output:
    max_instruction_tokens: 409
    max_output_tokens: 3330
    avg_instruction_tokens: 88.92
    avg_output_tokens: 433.95
    """