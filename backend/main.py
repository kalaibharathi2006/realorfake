from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os

app = FastAPI(title="Fake News/Deepfake Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from model.inference import predictor

@app.get("/")
def read_root():
    return {"message": "Deepfake Detector API is running"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = predictor.predict_image(file_path)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        # Generate appropriate message based on detection result
        if result["label"] == "Real":
            message = "Analysis complete. This content appears to be authentic with no manipulation detected."
        elif result["label"] == "Scam":
            message = "⚠️ Warning! This image matches a known scam pattern. Do not trust this content."
        else:  # Fake
            message = "Analysis complete. Potential manipulation or deepfake artifacts detected."
        
        return {
            "filename": file.filename,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "message": result.get("message", message)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = predictor.predict_video(file_path)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "filename": file.filename,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "message": "Video frame analysis complete. " + ("No deepfake patterns found." if result["label"] == "Real" else "Temporal inconsistencies detected.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
