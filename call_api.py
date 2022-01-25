import requests
import json

data_input = {
        "age":20,
        "workclass":"?",
        "fnlgt":133515,
        "education":"Some-college",
        "education-num":10,
        "marital-status":"Never-married",
        "occupation":"?",
        "relationship":"Own-child",
        "race":"White",
        "sex":"Female",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":15,
        "native-country":"France"
    }

#r = requests.post("http://127.0.0.1:8000/predict/", data=json.dumps(data_input))
#print(r.json())


r = requests.post('https://census-classification.herokuapp.com/predict/', data=json.dumps(data_input))

print("result:\n", r.json())
print("status code:",r.status_code)