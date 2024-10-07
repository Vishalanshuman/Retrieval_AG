import re
from fastapi import FastAPI
from pydantic import BaseModel
from RAG import rag_system

app = FastAPI()

def extract_after_helpful_answer(text):
    match = re.search(r"Helpful Answer:(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

class QuestionRequest(BaseModel):
    question: str


    
@app.post("/ask_question")
def get_answer(question:QuestionRequest):
    try:
        answer=rag_system.run_pipeline(question.question)
        answer=extract_after_helpful_answer(answer)
        print(answer)
        return {"answer":answer}
    except Exception as e:
        return {"error":e.__str__()}
