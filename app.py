import os, shutil, uuid, uvicorn
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from main import run_pipeline

# Paths
ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "outputs"
CONFIG = ROOT / "configs" / "court_config.json"

for d in [UPLOADS, RESULTS]: d.mkdir(exist_ok=True)

app = FastAPI()
jobs: Dict[str, dict] = {}

def process_job(job_id: str, src: str, dst: str):
    def update_p(p: float, msg: str):
        jobs[job_id].update({"progress": round(p * 100, 1), "message": msg})

    try:
        run_pipeline(
            video_path=src,
            output_path=dst,
            config_path=str(CONFIG),
            weights="models/best.pt",
            imgsz=480,
            clip_every=20,
            progress_callback=update_p
        )
        jobs[job_id]["done"] = True
    except Exception as e:
        jobs[job_id].update({"message": f"Error: {e}", "done": False})

@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    jid = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    
    src = UPLOADS / f"{jid}{ext}"
    dst = RESULTS / f"{jid}.mp4"
    
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    jobs[jid] = {
        "progress": 0,
        "message": "Initializing...",
        "done": False,
        "file": f"{jid}.mp4",
        "name": file.filename
    }
    
    bg.add_task(process_job, jid, str(src), str(dst))
    return {"task_id": jid}

@app.get("/status/{jid}")
async def get_status(jid: str):
    if jid not in jobs: raise HTTPException(404)
    return jobs[jid]

@app.get("/download/{jid}")
async def download(jid: str):
    job = jobs.get(jid)
    if not job or not job["done"]: raise HTTPException(404)
    
    path = RESULTS / job["file"]
    if not path.exists(): raise HTTPException(404)
        
    return FileResponse(path, filename=f"analyzed_{job['name']}", media_type="video/mp4")

@app.get("/view/{jid}")
async def view(jid: str):
    job = jobs.get(jid)
    if not job or not job["done"]: raise HTTPException(404)
    return FileResponse(RESULTS / job["file"])

@app.get("/", response_class=HTMLResponse)
async def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    print("\n" + "="*40)
    print("Basketball Analyzer Started")
    print("URL: http://localhost:8000")
    print("="*40 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
