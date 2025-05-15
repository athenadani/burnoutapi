from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

model_name = "athenadani/burnout-greekbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: InputText):
    if not input.text.strip():
        return {"error": "Empty input"}

    results = classifier(input.text)[0]
    label_map = {
        "LABEL_0": "Χαμηλή Εξουθένωση",
        "LABEL_1": "Μέτρια Εξουθένωση",
        "LABEL_2": "Υψηλή Εξουθένωση"
    }
    scores = {
        label_map[r["label"]]: round(r["score"] * 100, 2)
        for r in results
    }
    best = max(results, key=lambda x: x["score"])
    return {
        "category": label_map[best["label"]],
        "scores": scores
    }