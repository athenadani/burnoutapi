from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_name = "athenadani/burnout-greekbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

label_map = {
    "LABEL_0": "Χαμηλή Εξουθένωση",
    "LABEL_1": "Μέτρια Εξουθένωση",
    "LABEL_2": "Υψηλή Εξουθένωση"
}

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Burnout model is running."}

@app.post("/predict")
def predict_burnout(data: TextRequest):
    if not data.text.strip():
        return {"error": "empty input"}

    results = classifier(data.text, truncation=True)[0]
    scores = {}
    for res in results:
        label_name = label_map.get(res["label"], res["label"])
        scores[label_name] = round(res["score"] * 100, 2)

    best = max(results, key=lambda x: x["score"])
    best_label = label_map.get(best["label"], best["label"])
    return {"category": best_label, "scores": scores}