import os
import pickle

from flask import Flask, jsonify

app = Flask(__name__)
model = None
if os.path.exists("model_linear"):
    model = pickle.load(open("model_linear", "rb"))


@app.route("/price/<value>")
def predict(value):
    if model:
        return jsonify({"value": model(value)}), 200
    return jsonify({"error": "can't predict price. model don't exists."}), 500


if __name__ == "__main__":
    app.run()
