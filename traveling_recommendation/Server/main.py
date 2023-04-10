from flask import Flask, jsonify
from flask import request
import pandas as pd
from flask_cors import CORS
import pickle

app = Flask(__name__)

CORS(app)


@app.route("/plus", methods=["GET"])
def connect():
    data = {'name' : 'hello'}
    print(data)
    # 학습시킨 값을 return.
    return data


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5050')
