# Specify base image:tag (starting point for our image)
FROM python:latest

COPY requirements.txt requirements.txt
# Update OS
RUN apt-get update \
    # Install minimal Python3 version
    # && apt-get install -y --no-install-recommends python3 python3-pip \ 
    # Remove mirrors making minimal image
    && rm -rf /var/lib/apt/lists/* \
    # 
    && pip3 install -r requirements.txt

# Copy data from docker context (from current folder (first `.`) to a folder within the container (second `.`)) 

COPY dummy_train.py dummy_train.py 


# When we run container this will be the command run
ENTRYPOINT ["python3", "dummy_train.py"]