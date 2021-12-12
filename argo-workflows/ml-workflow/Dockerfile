FROM python:3.8

COPY argo-workflows/ml-workflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /code
COPY argo-workflows/ml-workflow .
ENTRYPOINT python