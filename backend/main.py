from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from autoencoder import train_autoencoder, get_num_epochs, get_reconstruction_path, get_losses
import os


app = FastAPI()
UPLOAD_DIR = "uploads"
RECON_DIR = "reconstructions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECON_DIR, exist_ok=True)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only; restrict for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    print(f"Received file: {file.filename}")  # <-- ADD THIS LINE
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    background_tasks.add_task(train_autoencoder, path)
    return {"message": "Training started!"}

@app.get("/epochs")
def num_epochs():
    # Returns number of epochs completed (after training)
    return {"num_epochs": get_num_epochs()}

@app.get("/reconstruction/{epoch}")
def get_reconstruction(epoch: int):
    # Returns reconstructed image at given epoch as file
    path = get_reconstruction_path(epoch)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/losses")
def losses():
    # Returns list of losses per epoch
    return {"losses": get_losses()}
