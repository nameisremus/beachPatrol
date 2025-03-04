# Download NLTK data (needed for article parsing)
FROM python:3.11-slim AS nltkdata

RUN apt-get update && apt-get install -y --no-install-recommends wget

# Install just enough Python packages so we can run `nltk.downloader`
RUN pip install --no-cache-dir nltk==3.8.1

# Create a folder for the data
RUN mkdir -p /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data

RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# 2) Builder stage
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry==1.5.1

RUN poetry config virtualenvs.in-project true

RUN poetry install --no-root --no-dev --no-ansi

COPY . .

# Final minimal stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash beachPatrol

WORKDIR /app

COPY --from=builder /app /app

# Copy the NLTK data we downloaded in nltkdata stage
COPY --from=nltkdata /usr/share/nltk_data /usr/share/nltk_data

ENV NLTK_DATA=/usr/share/nltk_data

ENV PATH="/app/.venv/bin:$PATH"

USER beachPatrol

CMD ["echo", "Container built. Use docker-compose up."]