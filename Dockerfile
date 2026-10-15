FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[full]'
COPY scripts ./scripts
COPY docs ./docs
EXPOSE 8000 8501 8765
CMD ["python", "-m", "secom_lab.demo"]
