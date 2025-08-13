from openai import OpenAI
from datetime import datetime
import json
import os
import random

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
# 只有一个工具暂时:获取当前时间
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {},
        },
    }
]

def get_current_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间是：{current_time}"

# 响应函数

def get_response(messages):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )
    return completion

def call_with_messages():
    print("\n")
    messages = [
        {
            "content": input(
                "请输入："
            ),  # 提问示例："现在几点了？" "一个小时后几点" "北京天气如何？"
            "role": "user",
        }
    ]
    print("-" * 50)

    i= 1
    first_response = get_response(messages)
    assistant_response = first_response.choices[0].message # 助手的第一轮响应

    print(f"第{i}轮对话的大模型输出的信息：{first_response} \n")

    # 处理可能的空内容
    if assistant_response.content is None:
        assistant_response.content = ""
    
    # 模型回复添加到message中
    messages.append(assistant_response)

    # 如果不需要调用工具，则直接返回最终答案
    if assistant_response.tool_calls is None:
        print(f"不需要调用工具，最终答案：{assistant_response.content}")
        return
    # 如果需要调用工具，则进行模型的多轮调用，直到模型判断无需调用工具
    while assistant_response.tool_calls is not None:
        # 工具调用标准格式
        tool_info = {
            "content": "",
            "role": "tool",
            "tool_call_id": assistant_response.tool_calls[0].id,
        }
        if assistant_response.tool_calls[0].function.name == "get_current_time":
            tool_info["content"] = get_current_time()
        else:
            print(f"未定义的工具调用：{assistant_response.tool_calls[0].function.name}")
            return
        
        tool_output = tool_info["content"]
        print(f"工具输出：{tool_output}")
        print("-" * 50)
        # 将工具的回复添加到对话历史
        messages.append(tool_info)

        # 再次调用模型，让它基于工具结果继续思考
        assistant_response = get_response(messages).choices[0].message
        if assistant_response.content is None:
            assistant_response.content = ""
        
        # 将模型的新回复也添加到对话历史
        messages.append(assistant_response)
        
        i += 1
        print(f"第{i}轮大模型输出信息：{assistant_response}\n")
    
    print(f"最终答案：{assistant_response.content}")

if __name__ == "__main__":
    call_with_messages()

