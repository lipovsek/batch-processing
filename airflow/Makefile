start:
	mkdir -p ./dags ./logs ./plugins && echo -e "AIRFLOW_UID=$(id -u)" > .env && docker-compose up
stop:
	docker-compose down -v --remove-orphans && sudo rm -r dags logs plugins .env
start_minikube:
	minikube_start
start_k8s: start_minikube
	helmfile apply
expose:
	kubectl port-forward svc/apache-airflow-webserver 8080:8080 --namespace apache-airflow
stop_k8s:
	minikube delete
