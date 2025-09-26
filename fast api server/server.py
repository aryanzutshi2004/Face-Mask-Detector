from fastapi import FastAPI, File, UploadFile
from PIL import Image 
from io import BytesIO
import uvicorn
# Assuming your model setup and prediction function are in this file:
from model_helper import give_prediction 

# Initialize the FastAPI application
app = FastAPI()

@app.post("/prediction")
# The function must be `async` when dealing with file I/O operations like `await image.read()`
async def give_output(image: UploadFile = File(...)):
    """
    Receives an image file, processes it for face mask prediction,
    and returns the predicted class.
    """
    
    # 1. Read the image file content into raw bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        return {"error": f"Failed to read image bytes: {e}"}, 400

    # 2. Convert raw bytes to a PIL Image object
    try:
        # Wrap the bytes in a memory buffer
        image_buffer = BytesIO(image_bytes)
        # PIL decodes the bytes into an RGB image object
        pil_image = Image.open(image_buffer)
    except Exception as e:
        return {"error": f"Could not decode image file (ensure it's a valid format): {e}"}, 400

    # 3. Get the prediction
    try:
        # Pass the PIL Image to your model helper function
        prediction = give_prediction(pil_image)
    except Exception as e:
        # Catch errors from the model itself (e.g., CUDA issues, shape mismatch)
        return {"error": f"Model prediction failed: {e}"}, 500
    
    return {"Prediction": prediction}

@app.get("/") 
def say_hello():
    """Simple health check endpoint."""
    return {"hello": "Face Mask Prediction API is running. POST an image to /prediction."}

# To run the server, save this code as `app.py` and run the following command 
# in your terminal (make sure to install `uvicorn` and `python-multipart`):
# uvicorn app:app --reload