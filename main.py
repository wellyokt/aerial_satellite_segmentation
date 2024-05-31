from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = load_model('models/satellite_standard_unet_100epochs_18May2024.hdf5')


def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  
    img = np.array(img) / 255.0  
    return img

IMAGE_SIZE = 256  

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()  
    img = preprocess_image(contents)  
    prediction = model.predict(np.expand_dims(img, axis=0))  
    predicted_img = np.argmax(prediction, axis=3)[0]  
    return predicted_img.tolist() 
