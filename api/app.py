from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
import io
from pathlib import Path, PosixPath
import pathlib
import platform

# Set platform-specific path handling
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="YOLOv5 Machine Learning API for CFU Counting Prediction",
    description="Apply a custom Yolov5 model on an image (jpg, png) and return the predicted positions of the CFU (JSON)",
    version="0.0.1",
)

# Define the directory of the application and the associated Unix-compatible paths
app_directory = PosixPath(Path(__file__).resolve().parent)
parent_directory = app_directory.parent
yolo_path = app_directory.parent / "yolov5"
model_path = app_directory.parent / "models" / "20240818_UFC_counting_model_v1.0.pt"

# Check if the YOLOv5 model exists, and raise an error if not found
if not model_path.exists():
    print(model_path)
    raise RuntimeError(f"Model not found at {model_path}")

# Load the YOLOv5 model
model = torch.hub.load(
    str(yolo_path), 
    'custom', 
    path=str(model_path), 
    skip_validation=True, 
    force_reload=True, 
    source='local'
)

# Root endpoint to welcome users to the API
@app.get("/")
async def home():
    """
    Root endpoint that returns a welcome message to perform health check of API routes.

    Returns:
        dict: A welcome message.
    """
    return {
        "message": "Welcome to the Custom YOLOv5 Machine Learning API!"
    }

# Endpoint to predict objects in an uploaded image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in an uploaded image using the YOLOv5 model.

    Args:
        file (UploadFile): The uploaded image file (png, jpg).

    Returns:
        JSONResponse: A JSON response containing the predictions.
    """
    try:
        # Read the uploaded image
        image = Image.open(io.BytesIO(await file.read()))
        
        # Perform the prediction using the YOLOv5 model
        results = model(image)

        # Process the results and convert to JSON format
        results_json = results.pandas().xyxy[0].to_json(orient="records")
        
        # Return the predictions as a JSON response
        return JSONResponse(content={"predictions": results_json})
    
    except UnidentifiedImageError:
        # Raise an error if the uploaded file is not a valid image
        raise HTTPException(status_code=400, detail="Invalid image format")