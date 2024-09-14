FROM python:3.11.7
RUN mkdir /app
COPY /src /app
COPY pyproject.toml /app
WORKDIR /app
ENV CFLAGS=-Qunused-arguments
ENV CPPFLAGS=-Qunused-arguments
RUN --mount=type=cache,target=/root/.cache/pip pip --timeout=1000 install .
RUN sudo apt install tesseract-ocr poppler-utils libmagic-dev
EXPOSE 8000
ENTRYPOINT ["studysync", "--port", "8000"]