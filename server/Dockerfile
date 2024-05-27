FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS cuda

# Install Conda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

WORKDIR /app

COPY requirements.txt requirements.txt

RUN conda update conda --yes \ 
    && conda config --add channels conda-forge \
    && conda create --name app_env python=3.9 \
    && conda install --yes --file requirements.txt

# Some packages are not available in conda
RUN pip install --upgrade pip
RUN pip install Flask-PluginKit
RUN pip install Flask-Session
RUN pip install is-safe-url
RUN pip install python3-discogs-client
RUN pip install cachelib
#RUN pip install lenskit
RUN pip install pytest
RUN pip install tensorflow==2.15.0 # needed for TF-recommenders (Matrix Factorization)
RUN pip install tensorflow-recommenders
# RUN pip install tensorflow[and-cuda]
RUN pip install black --find-links https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.12.1+cu113 torchaudio==0.12.1+cu113 torchvision==0.13.1+cu113
# ELSA https://github.com/recombee/ELSA
RUN pip install elsarec

# Add CUDA support
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update && apt-get install -y nvidia-container-toolkit
RUN nvidia-ctk runtime configure --runtime=docker

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.9/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.9/site-packages/nvidia/cuda_cupti/lib:/opt/conda/lib/python3.9/site-packages/nvidia/cuda_nvcc/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/python3.9/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.9/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.9/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/python3.9/site-packages/nvidia/cusparse/lib:/opt/conda/lib/python3.9/site-packages/nvidia/nccl/lib:/opt/conda/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

WORKDIR /app

EXPOSE 5000

# ENTRYPOINT [ "bash" ]
# CMD ["-c", "pip freeze"]

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "0", "--preload", "--log-level", "debug", "--access-logfile", "/app/access_logfile.txt", "--error-logfile", "/app/error_logfile.txt", "app:create_app()"]
# CMD ["tf_test.py"]
# CMD ["torch_test.py"]