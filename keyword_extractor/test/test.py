import requests
text="Looking for job in machine learning, where I can enhance my AI skills using ml"
response=requests.post("http://172.16.0.178:5000/keyword/extract",json={"text":text})
print(response.json())