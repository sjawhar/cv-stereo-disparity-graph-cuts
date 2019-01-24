# Stereo Disparity Maps

This repo contains a Python implementation of [Kolmogorov and Zabih’s Graph Cuts Stereo Matching Algorithm](https://hal-enpc.archives-ouvertes.fr/hal-01074878/document). This algorithm generates disparity maps from pairs of stereo images by posing the problem as one of finding the miniminum cut of a graph. Please refer to the [Jupyter notebook](./app/Report.ipynb) for a write-up. A video report is also available at [https://youtu.be/muDOTiv_v-8](https://youtu.be/muDOTiv_v-8).

The code for this implementation can be run using the provided Dockerfile and docker-compose.yml. Just run `docker-compose up` to build and run the container, then either:
1. Go to http://localhost:8888 to see the Jupyter notebook for this project; or
2. `docker exec` into the container and use the included CLI to run the disparity algorithm. You can run the SSD on all test images using `python proj.py -s all`. You can run the graphcut algorithm using `python proj.py -g $IMAGE_PAIR`, where `$IMAGE_PAIR` is the name of an image pair in the input directory. Run `python proj.py -h` for full usage instructions.


## Setup
g++ is required to install PyMaxflow, and libgtk is required for OpenCV. If not using the provided Dockerfile, please make sure these are installed on your system:
```bash
apt-get update
apt-get install -y g++ libgtk2.0-dev
conda env update -f environment.yml
```
