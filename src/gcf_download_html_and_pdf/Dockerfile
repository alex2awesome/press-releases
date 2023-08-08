FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update \
 && apt-get install -yq \
    sudo \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    gcc \
    g++ \
    libxml2-dev \
    libxslt1-dev  \
    antiword \
    unrtf \
    ffmpeg \
    libxext6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 webserver:server