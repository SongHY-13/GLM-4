# 本脚本用于测试my_demo中的各种需要调用的api和服务
import requests
import json
from LanchainClass import ChatGLM4_LLM

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt, "history": []}
    response = requests.post(url='http://0.0.0.0:8001', headers=headers, data=json.dumps(data))
    return response.json()['response']

def test_lanchain():
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    llm = ChatGLM4_LLM(mode_name_or_path="/home/data/GLM-4/modelTemp/glm-4-9b-chat", gen_kwargs=gen_kwargs)
    print(llm.invoke("你能完成投诉工单的投诉类型分析吗"))

if __name__ == '__main__':
    # print(get_completion('你好，讲个幽默小故事'))
    test_lanchain()