import requests

# API 地址
url = "http://127.0.0.1:5000/ask"

# 请求数据
data = {
    "question": "什么是人工智能?"
}

# 发送 POST 请求
response = requests.post(url, json=data)

# 打印返回的结果
if response.status_code == 200:
    print("回答：", response.json().get('answer'))
else:
    print("错误：", response.json().get('error'))
