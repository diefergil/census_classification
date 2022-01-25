import os
import sys
import json
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from fastapi.testclient import TestClient
# Import our app from main.py.
from main import app
from constants import positive_example, negative_example
# Instantiate the testing client with our app.
client = TestClient(app)


    

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    
def test_api_negative_example(negative_example):
    with TestClient(app) as client: # need when use startup and shutdown
        r = client.post("/predict/", data=json.dumps(negative_example))
        assert r.json() == {'prediction': '<=50K'}
    
def test_api_positive_example(positive_example):
    with TestClient(app) as client:
        r = client.post("/predict/", data=json.dumps(positive_example))
        assert r.json() == {'prediction': '>50K'}
    
# def test_get_path():
#     r = client.get("/items/42")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 1 of 42"}
# 
# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}
# 
# 
# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200