#!/bin/bash
set -o pipefail

# https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html#docker-compose-yaml
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.2/docker-compose.yaml'

curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.2/airflow.sh'
sudo chmod +x airflow.sh