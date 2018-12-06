FROM jupyter/scipy-notebook
RUN pip install \
        opencv-python==3.4.3.18 \
        pymaxflow==1.2.11

WORKDIR /home/jovyan
