import requests
text="Looking for job in machine learning, where I can enhance my AI skills"
response=requests.post("http://172.16.0.178:8002/keywords/extract",json={"text":text})
print(response.json())