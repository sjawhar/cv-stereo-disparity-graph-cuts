FROM continuumio/miniconda3
RUN apt-get update \
 && apt-get install -y \
        g++ \
        gcc \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY cv_proj.yml /
RUN conda env update -f cv_proj.yml

RUN adduser --gecos 'CS 6476' --disabled-password python
USER python
WORKDIR /app
