FROM google/cloud-sdk:latest
WORKDIR /sign

RUN apt update -y && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg libsm6 libxext6 apt-transport-https ca-certificates gnupg python3 python3-pip

ENV PYTHONPATH=/sign

COPY requirements.txt /sign/
RUN pip install --break-system-packages torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --break-system-packages -r requirements.txt

COPY . /sign

CMD ["python3", "app.py"]

