FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda/

RUN apt-get -y update
RUN apt-get -y install git vim curl

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

COPY ds_configs ds_configs

COPY src src
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
