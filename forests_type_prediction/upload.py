import requests

with open('train.csv','rb') as f:
	r = requests.post('https://colab.research.google.com/drive/1MGG440t65k8ODN4qjOkcB8wFi7zcGzPN#scrollTo=8YPnxpqBjN1Z', files={'train.csv': f})