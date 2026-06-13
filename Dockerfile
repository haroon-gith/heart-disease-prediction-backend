# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Requirements pehle copy karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki files copy karo
COPY main.py .
COPY KNN_heart.pkl .
COPY Scaler.pkl .
COPY Columns.pkl .

# Port open karo
EXPOSE 7860

# API run karo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
