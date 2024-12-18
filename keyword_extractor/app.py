from fastapi import FastAPI, Request,Response
from pydantic import BaseModel
from src.keywordmodel import KeywordExtractionTesting
model=KeywordExtractionTesting("models/kmeans.pkl","models/tfidf.pkl","models/umap.pkl")
app = FastAPI()
class KeywordExtraction(BaseModel):
    text: str
@app.get("/ping")
async def ping():
  return {"ping": "pong"}
@app.post("/keyword/extract")
async def exctract_keyword(data:KeywordExtraction):
  text=data.text
  keywords=model.process(text)
  print(f"keywords:{keywords}")
  return {"keyword":keywords}