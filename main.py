from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import databases
import sqlalchemy
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()
model = load_model('models/satellite_standard_unet_100epochs_18May2024.hdf5')

DATABASE_URL = "sqlite:///./predictions.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

predictions = sqlalchemy.Table(
    "predictions",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("input_data", sqlalchemy.String),
    sqlalchemy.Column("output_data", sqlalchemy.String),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
metadata.create_all(engine)

def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return img

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_img = np.argmax(prediction, axis=3)[0]
    
    # Convert input and output data to strings
    input_data_str = img.tobytes().hex()
    output_data_str = predicted_img.tobytes().hex()
    
    # Insert data into database
    query = predictions.insert().values(
        input_data=input_data_str,
        output_data=output_data_str,
        timestamp=datetime.utcnow()
    )
    await database.execute(query)
    
    return predicted_img.tolist()
