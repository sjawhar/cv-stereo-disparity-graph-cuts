# Stereo Disparity Maps
Included in this repo is a Dockerfile and docker-compose.yml. Just run `docker-compose up` to build and run the container, then go to http://localhost:8888 to see the Jupyter notebook for this project.

A small CLI is included to run the disparity algorithm. You can run the SSD on all test images using `python proj.py -s all`. You can run the graphcut algorithm using `python proj.py -g $IMAGE_PAIR`, where `$IMAGE_PAIR` is the name of an image pair in the input directory. Run `python proj.py -h` for full usage instructions.

Video report available at [https://youtu.be/muDOTiv_v-8](https://youtu.be/muDOTiv_v-8)
