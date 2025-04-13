#!/usr/bin/env bash
set -e

# Checking if Redis is reachable
 if (echo > /dev/tcp/"$REDIS_HOST"/"$REDIS_PORT") &>/dev/null; then
  exit 0
else
  echo "Redis unreachable"
  exit 1
fi