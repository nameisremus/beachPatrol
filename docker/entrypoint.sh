#!/usr/bin/env bash
set -euo pipefail

# Check Redis with exponential backoff
REDIS_HOST="${REDIS_HOST:-}"
REDIS_PORT="${REDIS_PORT:-6380}"

if [[ -n "$REDIS_HOST" ]]; then
  echo "[entrypoint.sh] Checking Redis at $REDIS_HOST:$REDIS_PORT..."
  attempt=1
  max_attempts=5
  delay=2
  while [ $attempt -le $max_attempts ]; do
    if (echo > /dev/tcp/"$REDIS_HOST"/"$REDIS_PORT") &>/dev/null; then
      echo "[entrypoint.sh] Redis is reachable!"
      break
    else
      echo "[entrypoint.sh] ($attempt/$max_attempts) Redis not ready, retrying in ${delay}s..."
      sleep $delay
      attempt=$(( attempt + 1 ))
      delay=$(( delay * 2 ))
    fi
  done

  if [ $attempt -gt $max_attempts ]; then
    echo "[entrypoint.sh] ERROR: Unable to connect to Redis at $REDIS_HOST:$REDIS_PORT"
    exit 1
  fi
fi

echo "[entrypoint.sh] All checks OK. Running main command."
exec "$@"