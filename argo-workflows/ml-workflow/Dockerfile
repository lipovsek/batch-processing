FROM python:3.8

COPY ml-workflow/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /code
COPY ml-workflow .
ENTRYPOINT python