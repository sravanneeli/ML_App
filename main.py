import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile

with open('./tf-models/imagenet1000_clsidx_to_labels.txt') as f:
    imagenet_id_to_label = eval(f.read())


def load_models():
    """
    load the models from disk
    and put them in a dictionary

    Returns:
        dict: loaded models
    """

    resnet50 = tf.saved_model.load('./tf-models/resnet50')

    models = {
        'resnet50': resnet50.signatures['serving_default']
    }

    return models


models_dict = load_models()


app = FastAPI()

@app.post("/image-classificaion/predict")
async def classify_image(file: UploadFile = File(...)):
    """
    Predict uploaded image class from the Imagenet Dataset.

    Args:
        file: File object (allowed extension: .jpg, jpeg, png)
    
    Return:
        prediction: Predicted class
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    if extension in ("jpg", "jpeg"):
        img = tf.io.decode_jpeg(await file.read())
    else:
        img = tf.io.decode_png(await file.read())
    img = tf.expand_dims(img, axis=0)
    prediction = models_dict['resnet50'](img)
    id = int(tf.argmax(prediction['probs'][0]).numpy())
    return {
        "class": imagenet_id_to_label[id]
    }

