FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
