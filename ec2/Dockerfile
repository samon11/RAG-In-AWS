FROM python:3.10-slim

RUN mkdir /app
WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

# Override the CMD to either api.py or run.py
CMD [ "python", "api.py"]