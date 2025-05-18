#!/bin/bash

# First make this script executable (chmod +x run_database_update.sh)
# Then run it with argumants. eg: ./run_database_update.sh update --years 2024 2025
# If qdrant is not already running (docker compose --env-file .env.compose up), it
# will be fired up. At the end, only selenium, chrome and database-service are
# stopped and removed, leaving qdrant running

# Forward all args to dataset-service
echo "Running database-service with arguments: $@"
docker compose --env-file .env.compose run --rm database-service "$@"

# Stop only chrome and selenium-hub (leave other services like qdrant running)
echo "Stopping chrome nodes and selenium-hub..."
docker compose --env-file .env.compose stop chrome selenium-hub database-service

# Remove chrome, selenium-hub and database-service
echo "Removing chrome, selenium-hub and database containers..."
docker compose --env-file .env.compose rm -f chrome selenium-hub database-service