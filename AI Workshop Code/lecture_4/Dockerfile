FROM astral/uv:python3.13-bookworm

WORKDIR /app

COPY pyproject.toml .

RUN uv venv
RUN uv sync

ENV PYTHONUNBUFFERED=1

