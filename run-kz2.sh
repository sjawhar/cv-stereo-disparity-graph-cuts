#!/bin/bash
set -euf -o pipefail

IMAGE_PAIR="${1:-}"
SEARCH_DEPTH="${2:-}"

if [ -z "${IMAGE_PAIR}" ] || [ -z "${SEARCH_DEPTH}" ]
then
    echo 'Usage: ./run-kz2.sh $IMAGE_PAIR $SEARCH_DEPTH'
    exit 1
fi

if [ ! -f bin/KZ2 ]
then
    echo 'KZ2 not found'
    exit 1
fi

input_dir="app/input/${IMAGE_PAIR}"
if [ ! -d "${input_dir}" ]
then
    echo "${IMAGE_PAIR} not found"
    ls app/input
    exit 1
fi

output_dir="app/output/${IMAGE_PAIR}"
mkdir -p "${output_dir}"

./bin/KZ2 "${input_dir}/im0.png" "${input_dir}/im1.png" "-${SEARCH_DEPTH}" 0 -o "${output_dir}/kz2.png" -p "${output_dir}/kz2.pfm"
