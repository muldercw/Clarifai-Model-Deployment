FROM nvcr.io/nvidia/tritonserver:23.03-py3

RUN apt-get update && \
    apt-get purge -y curl libcurl3-gnutls \
        libcurl4 libcurl4-openssl-dev && \
    apt-get install -y -q --no-install-recommends \
        libgfortran[3,5] g++ gcc openssl libpq-dev \
        libsndfile1 pkg-config make cmake \
        libsm6 libxrender-dev libgl1-mesa-glx ffmpeg

RUN pip install --upgrade pip \
    && pip install torch==2.0.1 \
    && pip install torchserve==0.9.0 \
    && pip install torchvision==0.18.0 \
    && pip install pillow==10.3.0 \
    && pip install numpy==1.24.2 \
    && pip install PyYAML==6.0.1

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q libglib2.0-0

EXPOSE 8000
EXPOSE 8001
CMD ["tritonserver", "--model-repository=/models", "--exit-on-error=false"]
