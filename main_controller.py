from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import cv2
import numpy as np

from pose_module import PoseModule
from face_module import FaceModule
from hand_module import HandModule
from surface_module import SurfaceModule  # ✅ added

app = FastAPI()

pose_mod = PoseModule()
face_mod = FaceModule()
hand_mod = HandModule()
surface_mod = SurfaceModule()  # ✅ added


def read_frame(contents: bytes):
    """Decode uploaded image bytes into RGB frame for MediaPipe / CV."""
    np_arr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bgr  # return both if needed


@app.get("/")
def health():
    return {"status": "ok", "endpoints": ["/face", "/pose", "/hand", "/surface"]}


@app.post("/pose")
async def detect_pose(file: UploadFile = File(...)):
    contents = await file.read()
    rgb, _ = read_frame(contents)
    return pose_mod.get_landmarks(rgb)


@app.post("/face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    rgb, _ = read_frame(contents)
    return face_mod.get_landmarks(rgb)


@app.post("/hand")
async def detect_hand(file: UploadFile = File(...)):
    contents = await file.read()
    rgb, _ = read_frame(contents)
    return hand_mod.get_landmarks(rgb)


# ✅ SURFACE ENDPOINT (NOTE: plane detection is Unity-side)
@app.post("/surface")
async def surface_info(file: UploadFile = File(...)):
    contents = await file.read()
    _, bgr = read_frame(contents)
    return surface_mod.get_surface_info(bgr)


if __name__ == "__main__":
    uvicorn.run("main_controller:app", host="0.0.0.0", port=8000, reload=False)
