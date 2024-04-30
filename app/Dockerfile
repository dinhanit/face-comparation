FROM python:3.7

RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    python3-dev python3-numpy \
    libgl1-mesa-glx  # This library provides libGL.so.1

WORKDIR /app

COPY . .

RUN pip install dlib
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python","api.py"]