FROM python:3.10-slim

WORKDIR /app

COPY fastapi_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl
COPY models/clfLR.pkl /app/models/clfLR.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
