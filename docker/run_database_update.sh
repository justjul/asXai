#!/bin/bash

# First make this script executable (chmod +x run_database_update.sh)
# Then run it with argumants. eg: ./run_database_update.sh update --years 2024 2025
# If qdrant is not already running (docker compose --env-file .env.compose up -d), it
# will be fired up. At the end, only selenium, chrome and database-update are
# stopped and removed, leaving qdrant running

# Forward all args to dataset-update 
echo "Running database-update with arguments: $@"
docker compose --env-file .env.compose run --rm database-update \
    python -m asxai.services.database.update "$@"

# Stop only chrome and selenium-hub (leave other services like qdrant running)
echo "Stopping chrome nodes and selenium-hub..."
docker compose --env-file .env.compose stop chrome selenium-hub database-update

# Remove chrome, selenium-hub and database-update
echo "Removing chrome, selenium-hub and database containers..."
docker compose --env-file .env.compose rm -f chrome selenium-hub database-update