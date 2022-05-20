from flask import Flask, render_template, request
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
import os
from dotenv import load_dotenv
import pickle
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torch import load as torch_load
from torch import device
from torch import max as torch_max
import re
import io
import onnx
import onnxruntime
import numpy as np

load_dotenv()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

app = Flask(__name__)

ort_session = onnxruntime.InferenceSession("model.onnx")

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

def model_predict(img):

    transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    x = transform(img).unsqueeze(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    outputs = ort_outs[0]

    print(outputs)
    preds = np.argmax(outputs)

    return preds

@app.route('/forward', methods=['POST'])
def forward():

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Get the image from post request
    try:
        img = Image.open(request.files['image'])
    except Exception as e:
        return "Bad format", 400
    print(img)

    # Make prediction
    #try:
    preds = model_predict(img)
    #except Exception as e:
    #    return "модель не смогла обработать данные", 403

    pred_class = classes[preds]

    result = str(pred_class)

    onnx_model = onnx.load("model.onnx")
    meta = onnx_model.metadata_props
        
    return render_template('index.html', prediction = result, meta = meta)