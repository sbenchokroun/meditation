FROM python:3.8.6-buster
COPY meditation /meditation
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD ["sh", "-c", "uvicorn meditation.api.fast:app --host 0.0.0.0 --port $PORT"]
