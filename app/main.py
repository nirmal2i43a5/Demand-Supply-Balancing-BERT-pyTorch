# main.py
import os
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ner_pipeline import (
    load_chunked_pipeline,
    predict_ner,
)

from utils.pdf_extract_utils import extract_text_from_pdf_bytes 


MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/biobert_ner_baseline_v1")


app = FastAPI(title="BIO NER API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class InText(BaseModel):
    text: str
    # max_tokens: int = 512
    # stride_tokens: int = 128




ner = load_chunked_pipeline( 
                             model_path=MODEL_PATH, aggregation_strategy="simple", device=None)


@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/predict")
def predict(payload: InText):
    text = (payload.text or "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")


    entities = predict_ner(
        ner, 
        text,
        max_tokens=512,
        stride_tokens=128,
    )
    return JSONResponse({"source": "text", 
                         "text": text, 
                         "entities": entities})


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    
    content = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf") or file.content_type == "application/pdf":
        text = extract_text_from_pdf_bytes(content)
        
    elif filename.endswith(".txt") or (file.content_type or "").startswith("text/"):
        text = content.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=415, detail="Upload a PDF or plain .txt")

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No readable text found.")

    entities = predict_ner(
        ner, 
        text,
        max_tokens=512,
        stride_tokens=128,
    )
    return JSONResponse({"source": "file", 
                         "filename": file.filename, 
                         "text": text, 
                         "entities": entities})
