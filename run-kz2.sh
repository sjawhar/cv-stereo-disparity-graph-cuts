#!/bin/bash
set -euf -o pipefail

IMAGE_PAIR="${1:-}"
MAX_DEPTH="${2:-}"

if [ -z "${IMAGE_PAIR}" ] || [ -z "${MAX_DEPTH}" ]
then
    echo 'Usage: ./run-kz2.sh $IMAGE_PAIR $MAX_DEPTH'
    exit 1
fi

if [ ! -f KZ2 ]
then
    echo 'KZ2 not found'
    exit 1
fi

input_dir="app/input-images/${IMAGE_PAIR}"
if [ ! -d "${input_dir}" ]
then
    echo "${IMAGE_PAIR} not found"
    ls app/input-images
    exit 1
fi

output_dir="app/output-images/${IMAGE_PAIR}"
mkdir -p "${output_dir}"

./KZ2 "${input_dir}/im0.png" "${input_dir}/im1.png" "-${MAX_DEPTH}" 0 -o "${output_dir}/kz2.png"
