from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
api_key = os.getenv('api_key')



def LLM(model, messages, tools=[]):
    client = ZhipuAI(api_key=api_key) 
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        tools=tools,
        tool_choice="auto",
        do_sample=False,
    )

    # 检查 function 是否存在
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        function = response.choices[0].message.tool_calls[0].function
        func_args = function.arguments
        func_name = function.name
        return func_name, json.loads(func_args)
    else:
        return response.choices[0].message.content

