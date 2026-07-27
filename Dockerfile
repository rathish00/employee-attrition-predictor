# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the model at image build time so the container starts ready to serve.
# (Swap in the real Kaggle CSV at data/WA_Fn-UseC_-HR-Employee-Attrition.csv
# before building if you want the image trained on real data.)
RUN python3 data/generate_data.py \
    && python3 notebooks/01_eda.py \
    && python3 notebooks/02_train_model.py

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
