FROM python:3.12.4

WORKDIR /app
COPY RAG .
COPY Myapp.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["uvicorn", "Myapp:app", "--host", "0.0.0.0", "--port", "8000"]
