from pathlib import Path

import spacy
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.cnn_model import WEIGHTS_PATH, load_model, predict
from helper_lib.generator import generate_png
from helper_lib.model import Generator
from helper_lib.utils import load_model as load_weights

app = FastAPI()

# Medium English model ships with 300-dimensional word vectors (Module 2)
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)

# CIFAR10 CNN classifier (Assignment 2); weights come from the training notebook
cnn = load_model() if WEIGHTS_PATH.exists() else None

# MNIST GAN generator (Assignment 3); weights come from the training notebook
GAN_WEIGHTS_PATH = Path(__file__).parent / "gan_mnist.pt"
gan_generator = load_weights(Generator(), GAN_WEIGHTS_PATH) if GAN_WEIGHTS_PATH.exists() else None


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def calculate_embedding(request: EmbeddingRequest):
    vector = nlp(request.word).vector
    return {"word": request.word, "dimensions": len(vector), "embedding": vector.tolist()}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if cnn is None:
        raise HTTPException(status_code=503, detail="Model weights not found")
    try:
        return predict(cnn, await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")


@app.get("/generate-digits")
def generate_digits(num_samples: int = Query(10, ge=1, le=64)):
    if gan_generator is None:
        raise HTTPException(status_code=503, detail="GAN weights not found")
    return Response(content=generate_png(gan_generator, num_samples), media_type="image/png")
