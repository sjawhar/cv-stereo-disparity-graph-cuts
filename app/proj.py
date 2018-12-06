from matplotlib import cm
from pfm import load_pfm
import argparse
import cv2
import numpy as np
import os
import stereo as st

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
REPORT_IMAGES = {
    'adirondack': (61, 80, 0.25),
    'cones': (65, 87, None),
    'flowers': (150, 203, 0.25),
    'motorcycle': (60, 118, 0.25),
    'pipes': (70, 82, 0.25),
}

def disparity_to_gray(disp, bgr=True):
    image = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
    is_occluded = disp < 0
    image[:] = np.where(is_occluded, 0, 255 * disp / disp.max())[:, :, np.newaxis]
    image[is_occluded] = [255, 255, 0] if bgr else [0, 255, 255]
    return image

def disparity_to_jet(disp, bgr=True):
    cm_jet = cm.ScalarMappable(cmap='jet')
    is_occluded = disp < 0
    jet = cm_jet.to_rgba(np.where(is_occluded, 0, disp), bytes=True)[:, :, :3]
    jet[is_occluded] = 0
    if not bgr:
        return jet
    return cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)

def get_accuracy(true, pred, scale):
    is_correct = np.abs(true - pred) <= 2. * scale
    total_accuracy = is_correct.sum() / float(true.size)
    is_visible = true >= 0
    visible_accuracy = is_correct[is_visible].sum() / float(is_visible.sum())
    return total_accuracy, visible_accuracy

def process_image_pair(image_pair, generate_ssd=False, generate_graphcut=False):
    search_depth, occlusion_cost, pfm_scale = REPORT_IMAGES[image_pair]
    left = cv2.imread(os.path.join(INPUT_DIR, image_pair, 'im0.png'))
    right = cv2.imread(os.path.join(INPUT_DIR, image_pair, 'im1.png'))

    disparity_ssd, accuracy_ssd, disparity_graphcut, accuracy_graphcut, ground_truth = None, None, None, None, None

    if pfm_scale:
        ground_truth = load_pfm(os.path.join(INPUT_DIR, image_pair, 'disp0.pfm'), pfm_scale)

    if generate_ssd:
        disparity_ssd = st.disparity(
            left, right,
            method=st.METHOD_SSD,
            search_depth=search_depth,
        )
        if pfm_scale is not None:
            accuracy_ssd = get_accuracy(ground_truth, disparity_ssd, pfm_scale)

    if generate_graphcut:
        disparity_graphcut = st.disparity(
            left, right,
            method=st.METHOD_GRAPHCUT,
            search_depth=search_depth,
            occlusion_cost=occlusion_cost,
        )
        if pfm_scale is not None:
            accuracy_ssd = get_accuracy(ground_truth, disparity_graphcut, pfm_scale)

    return disparity_ssd, accuracy_ssd, disparity_graphcut, accuracy_graphcut

if __name__ == '__main__':
    def main(image_pair, generate_ssd, generate_graphcut):
        disparity_ssd, accuracy_ssd, disparity_graphcut, accuracy_graphcut = process_image_pair(image_pair, args.ssd, args.graph_cut)
        for disparity, accuracy, method in [(disparity_ssd, accuracy_ssd, 'ssd'), (disparity_graphcut, accuracy_graphcut, 'graphcut')]:
            if disparity is None:
                continue
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, image_pair, '{}-gray.png'.format(method)),
                disparity_to_gray(disparity),
            )

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, image_pair, '{}-jet.png'.format(method)),
                disparity_to_jet(disparity),
            )

            np.save(
                os.path.join(OUTPUT_DIR, image_pair, '{}.npy'.format(method)),
                disparity,
            )
            if accuracy is not None:
                print(image_pair, method, 'accuracy', accuracy)

    parser = argparse.ArgumentParser(description = 'Generate disparity maps for Middlebury stereo image pairs')

    parser.add_argument('image_pairs', metavar='IMAGE_PAIRS', nargs='+', type=str, help='the name of the image pair in %s' % INPUT_DIR, choices=list(REPORT_IMAGES.keys()) + ['all'])
    parser.add_argument('-s', '--ssd', action='store_true', help='generate disparity map using ssd')
    parser.add_argument('-g', '--graph-cut', action='store_true', help='generate disparity map using graph cuts')

    args = parser.parse_args()

    if not (args.ssd or args.graph_cut):
        parser.error('No method selected. Add --ssd or --graph-cut')

    image_pairs = args.image_pairs[:]
    while len(image_pairs) > 0:
        image_pair = image_pairs.pop()
        if image_pair == 'all':
            image_pairs += list(REPORT_IMAGES.keys())
            continue
        main(image_pair, args.ssd, args.graph_cut)

