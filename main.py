from fastapi.middleware.cors import CORSMiddleware
# from keras.applications.mobilenet_v2 import preprocess_input
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import tensorflow as tf
import shutil
import os
# Making sure GPU is not used for inferring
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plastic_model = tf.keras.models.load_model(
    "saved_models/Plastic Classifier/model.h5", compile=False)

trash_model = tf.keras.models.load_model(
    "saved_models/Trash Classifier/model.h5", compile=False)

wadaba_classnames = [
    "01_PET",
    "02_PEHD",
    "03_PVC",
    "04_PELD",
    "05_PP",
    "06_PS"
]

hhg_classnames = [
    'battery',
    'cardboard',
    'clothes',
    'glass',
    'medical',
    'metal',
    'organic',
    'paper',
    'plastic',
    'shoes'
]

def preprocess_file_to_image(file):
    file_path = f"saved_images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load image as PIL image
    image = tf.keras.utils.load_img(file_path, target_size=(150, 150))
    # Expand dimension of image in an array because model expects multiple images in an array.
    image = np.expand_dims(image, axis=0)
    # Preprocess input with a value range of 1-255 to 0-1
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # Stack array sequentially
    image = np.vstack([image])

    return image, file_path  # Return the image and file path

@app.post("/trash-predict")
async def predict_trash(file: UploadFile = File(...)):
    # Check if the uploaded file has an allowed content type
    allowed_content_types = ["image/jpeg", "image/png"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415, detail="Invalid file format. Only JPEG and PNG images are allowed.")

    # Check if the file has an allowed file extension
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=415, detail="Invalid file format. Only JPEG and PNG images are allowed.")

    # Preprocess file to image array
    image, file_path = preprocess_file_to_image(file)

    # Trash type prediction
    assert trash_model is not None
    trash_pred_prob = trash_model.predict(image)
    trash_pred_class = int(trash_pred_prob.argmax(axis=-1))
    trash_pred_classname = hhg_classnames[trash_pred_class]
    trash_pred_confidence = round(100 * (np.max(trash_pred_prob[0])), 2)

    data = {
        "class": str(trash_pred_classname),
        "confidence": trash_pred_confidence
    }

    if trash_pred_classname == 'plastic':
        # Plastic type prediction
        assert plastic_model is not None
        plastic_pred_prob = plastic_model.predict(image)
        plastic_pred_class = int(plastic_pred_prob.argmax(axis=-1))
        plastic_pred_classname = wadaba_classnames[plastic_pred_class]
        plastic_pred_confidence = round(
            100 * (np.max(plastic_pred_prob[0])), 2)
        data["plastic_type"] = str(plastic_pred_classname)
        data["plastic_pred_confidence"] = plastic_pred_confidence

    # Delete the image file
    os.remove(file_path)

    return JSONResponse(content=data)


@app.post("/plastic-predict")
async def predict_plastic(file: UploadFile = File(...)):
    # Check if the uploaded file has an allowed content type
    allowed_content_types = ["image/jpeg", "image/png"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415, detail="Invalid file format. Only JPEG and PNG images are allowed.")

    # Check if the file has an allowed file extension
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=415, detail="Invalid file format. Only JPEG and PNG images are allowed.")

    # Preprocess file to image array
    image, file_path = preprocess_file_to_image(file)

    assert plastic_model is not None

    # Model prediction
    plastic_pred_prob = plastic_model.predict(image)
    plastic_pred_class = int(plastic_pred_prob.argmax(axis=-1))
    plastic_pred_classname = wadaba_classnames[plastic_pred_class]
    plastic_pred_confidence = round(100 * (np.max(plastic_pred_prob[0])), 2)

    data = {
        "class": str(plastic_pred_classname),
        "confidence": plastic_pred_confidence }

    # Delete the image file
    os.remove(file_path)

    return JSONResponse(content=data)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
