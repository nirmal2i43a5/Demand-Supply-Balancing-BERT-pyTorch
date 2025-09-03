from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.ner import load_pipeline, predict_ner
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

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


ner = load_pipeline()


@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/predict")
def predict(request: Request, payload: InText):
    print("Received text:", payload.text)
    entities  = predict_ner(ner, payload.text)
    print("Predicted entities:----------------------------------", entities)
    return JSONResponse(entities )
