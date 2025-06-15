# Download NLTK data (needed for article parsing)
FROM python:3.13.3-slim AS nltkdata

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Installing enough Python packages so we can run `nltk.downloader`
RUN pip install --no-cache-dir nltk==3.8.1

# Create a folder for the data
RUN mkdir -p /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

# 2) Builder stage
FROM python:3.13.3-slim AS builder

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:5.1.6-0+deb12u1 \
    build-essential=12.9 \
    gfortran=4:12.2.0-3 \
    python3-dev=3.11.2-1+b1 \
    libatlas-base-dev=3.10.3-13 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry==2.1.3 \
&& poetry config virtualenvs.create false

RUN poetry install --only main --no-root --no-interaction --no-ansi
COPY . .
RUN poetry install --only main --no-interaction --no-ansi

# discard build-only packages to slim back down the builder image
RUN apt-get purge -y --auto-remove \
    build-essential gfortran python3-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Final minimal stage
FROM python:3.13.3-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:5.1.6-0+deb12u1 \
    libatlas3-base=3.10.3-13 \
    libgfortran5=12.2.0-14+deb12u1 && \
    rm -rf /var/lib/apt/lists/*

# Copy **all** the system-installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin

# Create a non-root user
RUN useradd -ms /bin/bash beachPatrol

WORKDIR /app

COPY --from=builder /app /app

# Copy the NLTK data we downloaded in nltkdata stage
COPY --from=nltkdata /usr/share/nltk_data /usr/share/nltk_data

ENV NLTK_DATA=/usr/share/nltk_data

COPY --from=builder /app /app

# Copy entrypoint & healthcheck scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

RUN chown -R beachPatrol:beachPatrol /app
USER beachPatrol

# Healthcheck. (30s interval, 10s timeout)
HEALTHCHECK --interval=30s --timeout=10s CMD ["/usr/local/bin/healthcheck.sh"]

# Entrypoint script usage
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["echo", "Container built. Use docker-compose up."]