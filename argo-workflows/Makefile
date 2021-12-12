minikube_start:
	minikube start
start: minikube_start
	helm repo add argo https://argoproj.github.io/argo-helm && helm repo update && helm install argo-workflows argo/argo-workflows -n argo-workflows --create-namespace -f values.yaml
expose:
	k port-forward -n argo-workflows service/argo-workflows-server 2746
minikube_stop:
	minikube delete
stop: minikube_stop
build_workflow_image:
	docker build -f ml-workflow/Dockerfile  -t ${IMAGE:-ghcr.io/lipovsek/batch-processing/argo} .