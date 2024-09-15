FROM python:3.11-slim
RUN mkdir /app
COPY /src /app
COPY pyproject.toml /app
WORKDIR /app
RUN pip install .
RUN apt update
RUN apt -y install tesseract-ocr poppler-utils libmagic-dev
EXPOSE 8000
ENTRYPOINT ["studysync", "--port", "8000"]
