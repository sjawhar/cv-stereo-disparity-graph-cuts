import maxflow
import numpy as np
import sys
from .utils import to_gray

def disparity(image_left, image_right, **kwargs):
    solver = GraphCutDisparitySolver(image_left, image_right, **kwargs)
    return solver.solve()

# Based on https://github.com/pmonasse/disparity-with-graph-cuts
class GraphCutDisparitySolver:
    LABEL_OCCLUDED = 1

    NODE_ALPHA = -1
    NODE_ABSENT = -2

    IS_NODE = lambda x: x >=0

    def __init__(
        self,
        image_left,
        image_right,
        always_randomize=False,
        search_depth=30,
        max_levels=-1,
        max_iterations=4,
        occlusion_cost=-1,
        smoothness_cost_high=-1,
        smoothness_cost_low=-1,
        smoothness_threshold=8,
    ):
        # TODO: Validate params
        self.always_randomize = always_randomize
        self.max_levels = search_depth if max_levels < 0 else max_levels
        self.max_iterations = max_iterations
        self.occlusion_cost = occlusion_cost
        self.smoothness_cost_low = smoothness_cost_low if smoothness_cost_low > 0 else 0.2 * self.occlusion_cost
        self.smoothness_cost_high = smoothness_cost_high if smoothness_cost_high > 0 else 3 * self.smoothness_cost_low
        self.smoothness_threshold = smoothness_threshold

        self.image_left = to_gray(image_left)
        self.image_right = to_gray(image_right)
        self.image_shape = self.image_left.shape
        self.image_size = self.image_left.size
        self.image_indices = np.indices(self.image_shape)
        self.energy = float('inf')

        search_interval = (search_depth // self.max_levels) + bool(search_depth % self.max_levels)
        self.search_levels = -1 * np.arange(0, search_depth + 1, search_interval)[::-1]
        rank = np.empty(len(self.search_levels), dtype=int)
        rank[np.argsort(self.search_levels)] = np.arange(len(self.search_levels))
        self.label_rank = dict(zip(self.search_levels, rank))

        self.build_neighbors()

    def is_in_image(self, x):
        return (0 <= x) & (x < self.image_shape[1])

    def build_neighbors(self):
        indices = np.indices(self.image_shape)

        neighbors_one_p = indices[:, 1:, :].reshape(2, -1)
        neighbors_one_q = neighbors_one_p + [[-1],[0]]
        neighbors_two_p = indices[:, :, :-1].reshape(2, -1)
        neighbors_two_q = neighbors_two_p + [[0],[1]]

        self.neighbors = np.array([
            np.concatenate([neighbors_one_p, neighbors_two_p], axis=1),
            np.concatenate([neighbors_one_q, neighbors_two_q], axis=1),
        ])
        self.neighbors_rolled = list(np.rollaxis(self.neighbors, 1))

        indices_p, indices_q = self.neighbors
        diff_left = self.image_left[list(indices_p)] - self.image_left[list(indices_q)]
        self.is_left_under = np.abs(diff_left) < self.smoothness_threshold

    def solve(self):
        self.labels = np.full(self.image_shape, self.LABEL_OCCLUDED, dtype=np.int)
        label_done = np.zeros(len(self.search_levels), dtype=bool)

        for i in range(self.max_iterations):
            if i == 0 or self.always_randomize:
                label_order = np.random.permutation(self.search_levels)

            for label in label_order:
                # print('iteration', i, 'label', label)
                label_index = self.label_rank[label]
                if label_done[label_index]:
                    continue

                is_expanded = self.expand_label(label)
                if is_expanded:
                    label_done[:] = False
                label_done[label_index] = True

            if label_done.all():
                break

        return -1 * self.labels

    def expand_label(self, label):
        is_expanded = False
        g = maxflow.Graph[int](2*self.image_size, 12*self.image_size)
        # print('Adding data+occlusion terms for', label)
        self.add_data_occlusion_terms(g, label)
        # print('Adding smoothness terms for', label)
        self.add_smoothness_terms(g, label)
        # print('Adding uniqueness terms for', label)
        self.add_uniqueness_terms(g, label)

        energy = g.maxflow() + self.e_data_occlusion
        if energy < self.energy:
            # print('new energy', energy, 'updating labels', label)
            self.update_labels(g, label)
            is_expanded = True
        self.energy = energy
        return is_expanded

    def add_data_occlusion_terms(self, g, label):
        indices_y, indices_x = self.image_indices
        is_label = self.labels == label
        is_occluded = self.labels == self.LABEL_OCCLUDED

        indices_shifted = np.where(is_occluded, indices_x, indices_x + self.labels)
        # assert self.is_in_image(indices_shifted[np.logical_not(is_occluded)]).all()
        ssd_active = np.square(self.image_left - self.image_right[indices_y, indices_shifted]) - self.occlusion_cost
        ssd_active[is_occluded | is_label] = -self.occlusion_cost - 1
        nodes_active = np.zeros(self.image_shape, dtype=np.int)
        nodes_active[is_occluded] = self.NODE_ABSENT
        nodes_active[is_label] = self.NODE_ALPHA
        is_node_active = np.logical_not(is_label | is_occluded)
        e_data_occlusion = ssd_active[is_label].sum()

        is_occluded = np.logical_not(self.is_in_image(indices_x + label))
        indices_shifted = np.where(is_occluded, indices_x, indices_x + label)
        ssd_label = np.square(self.image_left - self.image_right[indices_y, indices_shifted]) - self.occlusion_cost
        ssd_label[is_occluded | is_label] = -self.occlusion_cost - 1
        nodes_label = np.zeros(self.image_shape, dtype=np.int)
        nodes_label[is_occluded] = self.NODE_ABSENT
        nodes_label[is_label] = self.NODE_ALPHA
        is_node_label = np.logical_not(is_label | is_occluded)

        num_nodes = is_node_label.sum() + is_node_active.sum()
        node_ids = g.add_nodes(num_nodes)
        node_index = 0
        for row, col in np.ndindex(self.image_shape):
            if is_node_active[row, col]:
                node_id = node_ids[node_index]
                nodes_active[row, col] = node_id
                node_index += 1
                cost_active = ssd_active[row, col]
                # assert cost_active >= -self.occlusion_cost
                g.add_tedge(node_id, 0, cost_active)

            if is_node_label[row, col]:
                node_id = node_ids[node_index]
                nodes_label[row, col] = node_id
                node_index += 1
                cost_label = ssd_label[row, col]
                # assert cost_label >= -self.occlusion_cost
                g.add_tedge(node_id, cost_label, 0)

        # assert node_index == num_nodes

        self.is_node_active = is_node_active
        self.is_node_label = is_node_label
        self.nodes_active = nodes_active
        self.nodes_label = nodes_label
        self.e_data_occlusion = e_data_occlusion

    def add_smoothness_terms(self, g, label):
        labels_p, labels_q = self.labels[self.neighbors_rolled]

        penalty_label = self.get_smoothness_penalty(label)
        penalty_active_p = self.get_smoothness_penalty(labels_p)
        penalty_active_q = self.get_smoothness_penalty(labels_q)

        indices_p, indices_q = self.neighbors
        is_p_in_range = self.is_in_image(indices_p[1, :] + labels_q)
        is_q_in_range = self.is_in_image(indices_q[1, :] + labels_p)

        for neighbor_index in range(self.neighbors.shape[2]):
            indices_y, indices_x = self.neighbors.T[neighbor_index]
            label_p, label_q = self.labels[indices_y, indices_x]
            node_l_p, node_l_q = self.nodes_label[indices_y, indices_x]
            node_a_p, node_a_q = self.nodes_active[indices_y, indices_x]
            is_p_active, is_q_active = self.is_node_active[indices_y, indices_x]

            if node_l_p != self.NODE_ABSENT and node_l_q != self.NODE_ABSENT:
                penalty = penalty_label[neighbor_index]
                # assert penalty > 0
                if node_l_p != self.NODE_ALPHA and node_l_q != self.NODE_ALPHA:
                    self.add_smoothness_weights(g, node_l_p, node_l_q, 0, penalty, penalty, 0)
                elif node_l_p != self.NODE_ALPHA:
                    g.add_tedge(node_l_p, 0, penalty)
                elif node_l_q != self.NODE_ALPHA:
                    g.add_tedge(node_l_q, 0, penalty)

            penalty_p, penalty_q = penalty_active_p[neighbor_index], penalty_active_q[neighbor_index]

            if label_p == label_q:
                if not is_p_active or not is_q_active:
                    continue
                # assert label_p != label and label_p != self.LABEL_OCCLUDED
                # assert penalty_p > 0
                self.add_smoothness_weights(g, node_a_p, node_a_q, 0, penalty_p, penalty_p, 0)
                continue

            if is_p_active and is_q_in_range[neighbor_index]:
                # assert penalty_p > 0
                g.add_tedge(node_a_p, 0, penalty_p)

            if is_q_active and is_p_in_range[neighbor_index]:
                # assert penalty_q > 0
                g.add_tedge(node_a_q, 0, penalty_q)

    def _shift(self, indices, shift):
        _, width = self.image_shape
        indices_shifted = np.copy(indices)
        indices_shifted[1, :] += shift
        is_in_image = self.is_in_image(indices_shifted[1, :])
        indices_shifted[1, :] = np.clip(indices_shifted[1, :], 0, width - 1)
        return indices_shifted, is_in_image

    def get_smoothness_penalty(self, labels):
        indices_p, indices_q = self.neighbors
        if type(labels) is np.ndarray:
            labels = labels[self.is_left_under]

        smoothness = np.full(indices_p.shape[1], self.smoothness_cost_low, dtype=np.float)

        indices_p_shifted, is_p_in_image = self._shift(indices_p[:, self.is_left_under], labels)
        indices_q_shifted, is_q_in_image = self._shift(indices_q[:, self.is_left_under], labels)
        diff_right = self.image_right[list(indices_p_shifted)] - self.image_right[list(indices_q_shifted)]

        is_left_under = np.copy(self.is_left_under)
        is_left_under[is_left_under] = np.abs(diff_right) < self.smoothness_threshold
        smoothness[is_left_under] = self.smoothness_cost_high

        is_left_under[:] = self.is_left_under
        is_left_under[is_left_under] = np.logical_not(is_p_in_image & is_q_in_image)
        smoothness[is_left_under] = 0

        return smoothness

    def add_smoothness_weights(self, g, node1, node2, w1, w2, w3, w4):
        w0 = w1 - w2
        g.add_tedge(node1, w4, w2)
        g.add_tedge(node2, 0, w0)
        g.add_edge(node1, node2, 0, w3 - w4 - w0)

    def add_uniqueness_terms(self, g, label):
        # assert (self.labels[self.is_node_active] != self.LABEL_OCCLUDED).all()

        _, width = self.image_shape
        indices_y, indices_x = self.image_indices
        indices_shifted = indices_x + self.labels - label
        is_shift_valid = self.is_in_image(indices_shifted)
        indices_shifted = np.clip(indices_shifted, 0, width - 1)
        forbid = self.is_node_active & is_shift_valid
        forbid_label = self.nodes_label[indices_y, indices_shifted][forbid]
        forbid_active = self.nodes_active[forbid]
        self.add_uniqueness_weights(g, forbid_active, forbid_label)
        # assert (forbid_label >= 0).all()
        # assert (forbid_active >= 0).all()

        is_node_label = self.nodes_label != self.NODE_ABSENT
        forbid = self.is_node_active & is_node_label
        self.add_uniqueness_weights(g, self.nodes_active[forbid], self.nodes_label[forbid])
        # assert (self.nodes_label[forbid] >= 0).all()
        # assert (self.nodes_active[forbid] >= 0).all()

    def add_uniqueness_weights(self, g, sources, targets):
        for i in range(sources.size):
            g.add_edge(sources[i], targets[i], sys.maxsize, 0)

    def update_labels(self, g, label):
        is_node_active = np.copy(self.is_node_active)
        if is_node_active.any():
            nodes_active = self.nodes_active[is_node_active]
            is_node_active[is_node_active] = g.get_grid_segments(nodes_active)
            self.labels[is_node_active] = self.LABEL_OCCLUDED

        is_node_label = np.copy(self.is_node_label)
        if is_node_label.any():
            nodes_label = self.nodes_label[is_node_label]
            is_node_label[is_node_label] = g.get_grid_segments(nodes_label)
            self.labels[is_node_label] = label
