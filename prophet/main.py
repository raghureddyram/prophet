import os
import openai
from dotenv import load_dotenv
from .models import AnswerGenerator, LoadIndex
from fastapi import FastAPI
import uvicorn
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("prophet.main:app", host="0.0.0.0", port=8000, reload=True)


@app.get("/earnings/{year}/answers")
async def get_answer(year: str, question: str):
    faiss_index, pre_processor, paragraphs  = LoadIndex(year).load_index()
    encoder_model = pre_processor.encoder_model
    answer_generator = AnswerGenerator(encoder_model=encoder_model, index=faiss_index, paragraphs=paragraphs)
    answer = answer_generator.generate_answer(question)

    return {"answer": answer}