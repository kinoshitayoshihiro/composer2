FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir '.[realtime]'
EXPOSE 8000
CMD ["python", "-m", "realtime.phrase_ws"]
