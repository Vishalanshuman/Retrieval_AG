import re
from fastapi import FastAPI
from pydantic import BaseModel
from RAG import rag_system

app = FastAPI()

def extract_after_helpful_answer(text: str) -> str:
    keyword = "Helpful Answer"
    if keyword in text:
        return text.split(keyword, 1)[1].strip()
    return ""

class QuestionRequest(BaseModel):
    question: str


    
@app.post("/ask_question")
def get_answer(question:QuestionRequest):
    try:
        answer=rag_system.run_pipeline(question.question)
        print(answer)
        answer=extract_after_helpful_answer(answer)[1:]
        return {"answer":answer}
    except Exception as e:
        return {"error":e.__str__()}
