FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer -- Docker caches this,
# so rebuilds are fast unless requirements.txt actually changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY app/ ./app/
COPY streamlit_app.py .

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides this
# for the streamlit service specifically
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
