from PIL import Image
from flask import Flask, request, Response
import jsonpickle
import numpy as np

from mobilenet import Mobilenet
from utils import decode_image_request_to_ndarray

app = Flask(__name__)


@app.route('/mobilenet/file', methods=['POST'])
def predict():
    img = decode_image_request_to_ndarray(request)

    results = Mobilenet().predict(img)

    response = {
        'prediction_results': results
    }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route("/mobilenet/multipart", methods=["POST"])
def home():
    image = Image.open(request.files['image'])
    image_array = np.array(image)

    results = Mobilenet().predict(image_array)

    response = {
        'prediction_results': results
    }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=5000)
