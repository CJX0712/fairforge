FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fairforge/ ./fairforge/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --no-deps . \
    && python -m pytest -q -W ignore::UserWarning

CMD ["python", "examples/run_demo.py"]
