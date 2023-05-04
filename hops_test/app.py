# import the necessary packages
import keras.utils as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet50
import numpy as np
import cv2

from flask import Flask
import ghhops_server as hs

import rhino3dm

#register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/pointat",
    name="PointAt",
    description="Get point along curve",
    icon="test_icon.png",
    inputs=[
        hs.HopsCurve("Curve","C","Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on curve to evaluate"),
    ],
    outputs=[
        hs.HopsPoint("p", "p", "Point on curve at t")
    ],
)

def pointat(curve: rhino3dm.Curve, t):
    return curve.PointAt(t)

@hops.component(
    "/resnet",
    name="ResNet",
    description="Predict image with ResNet",
    icon="",
    inputs=[
        hs.HopsString("Path", "path", "Path to image")
    ],
    outputs=[
        hs.HopsString("Prediction", "x", "Prediction")
    ],
) 

def resnet(path):
    orig = cv2.imread(path)
    image = image_utils.load_img(path, target_size=(224, 224))
    image = image_utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    model = ResNet50(weights="imagenet")

    preds = model.predict(image)
    P = decode_predictions(preds)

    predictions = []
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        predictions.append("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    return predictions[0]

if __name__ == "__main__":
    app.run()



