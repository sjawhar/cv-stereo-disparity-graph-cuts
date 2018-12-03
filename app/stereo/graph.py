import maxflow
import numpy as np
from .utils import to_gray

def disparity(image_left, image_right, **kwargs):
    solver = GraphCutDisparitySolver(image_left, image_right, **kwargs)
    return solver.solve()

# Based on https://github.com/pmonasse/disparity-with-graph-cuts
class GraphCutDisparitySolver:
    LABEL_OCCLUDED = -1

    NODE_ALPHA = -1
    NODE_ABSENT = -2

    IS_NODE = lambda x: x >=0

    def __init__(
        self,
        image_left,
        image_right,
        always_randomize=False,
        max_levels=16,
        max_iterations=4,
        occlusion_cost=-1,
        smoothness_cost_high=-1,
        smoothness_cost_low=-1,
        smoothness_threshold=8,
    ):
        # TODO: Validate params
        self.always_randomize = always_randomize
        self.max_levels = max_levels
        self.max_iterations = max_iterations
        self.occlusion_cost = occlusion_cost
        self.smoothness_cost_high = smoothness_cost_high
        self.smoothness_cost_low = smoothness_cost_low
        self.smoothness_threshold = smoothness_threshold

        self.image_left = to_gray(image_left)
        self.image_right = to_gray(image_right)
        self.image_shape = self.image_left.shape
        self.image_size = self.image_left.size
        self.image_indices = np.indices(self.image_shape)
        self.energy = float('inf')

        self.build_neighbors()

    def build_neighbors(self):
        indices = np.indices(self.image_shape)

        neighbors_one_p = indices[:, :, 1:].reshape(2, -1)
        neighbors_one_q = neighbors_one_p + [[0],[-1]]
        neighbors_two_p = indices[:, :-1, :].reshape(2, -1)
        neighbors_two_q = neighbors_two_p + [[1],[0]]

        self.neighbors = np.array([
            np.concatenate([neighbors_one_p, neighbors_two_p], axis=1),
            np.concatenate([neighbors_one_q, neighbors_two_q], axis=1),
        ])
        self.neighbors_rolled = list(np.rollaxis(self.neighbors, 1))
        indices_p, indices_q = self.neighbors

        diff_left = self.image_left[list(indices_p)] - self.image_right[list(indices_q)]
        self.is_left_under = np.abs(diff_left) < self.smoothness_threshold

    def solve(self):
        labels = np.full(self.image_shape, self.LABEL_OCCLUDED, dtype=np.int)
        label_done = np.zeros(self.max_levels, dtype=bool)

        for i in range(self.max_iterations):
            if (i == 0 or self.always_randomize):
                label_order = np.random.permutation(self.max_levels)

            for label in label_order[np.logical_not(label_done)]:
                is_expanded = self.expand_label(labels, label)

                if is_expanded:
                    label_done[:] = False

                label_done[label] = True

            if label_done.all():
                break

        self.labels = labels
        return labels

    def expand_label(self, labels, label):
        is_expanded = False
        g = maxflow.Graph[int](2*self.image_size, 12*self.image_size)
        self.add_data_occlusion_term(g, labels, label)
        self.add_smoothness_term(g, labels, label)
        self.add_uniqueness_term(g, labels, label)

        energy = g.maxflow()
        if (energy < self.energy):
            self.update_labels(g, labels, label)
            is_expanded = True
        self.energy = energy
        return is_expanded

    def add_data_occlusion_term(self, g, labels, label):
        _, width = self.image_shape
        indices_y, indices_x = self.image_indices
        is_label = labels == label
        is_occluded = labels == self.LABEL_OCCLUDED

        indices_shifted = np.where(is_occluded, indices_x, indices_x + labels)
        ssd_label = np.square(self.image_left - self.image_right[indices_y, indices_shifted]) - self.occlusion_cost
        nodes_label = np.zeros(self.image_shape, dtype=np.int)
        nodes_label[is_occluded] = self.NODE_ABSENT
        nodes_label[is_label] = self.NODE_ALPHA
        is_node_label = np.logical_not(is_label | is_occluded)
        e_data_occlusion = ssd_label[is_label].sum()

        is_occluded = is_occluded | (indices_x >= width - label)
        indices_shifted = np.where(is_occluded, indices_x, indices_x + label)
        ssd_active = np.square(self.image_left - self.image_right[indices_y, indices_shifted]) - self.occlusion_cost
        nodes_active = np.zeros(self.image_shape, dtype=np.int)
        nodes_active[is_occluded] = self.NODE_ABSENT
        nodes_active[is_label] = self.NODE_ALPHA
        is_node_active = np.logical_not(is_label | is_occluded)

        num_nodes = is_node_label.sum() + is_node_active.sum()
        node_ids = g.add_nodes(num_nodes)

        node_index = 0
        for row, col in np.ndindex(self.image_shape):
            if (nodes_label[row, col] == 0):
                node_id = node_ids[node_index]
                g.add_tedge(node_id, ssd_label[row, col], 0)
                nodes_label[row, col] = node_id
                node_index += 1

            if (nodes_active[row, col] == 0):
                node_id = node_ids[node_index]
                g.add_tedge(node_id, 0, ssd_active[row, col])
                nodes_active[row, col] = node_id
                node_index += 1

        self.is_node_active = is_node_active
        self.is_node_label = is_node_label
        self.nodes_active = nodes_active
        self.nodes_label = nodes_label

    def add_smoothness_term(self, g, labels, label):
        _, width = self.image_shape
        labels_p, labels_q = labels[self.neighbors_rolled]

        penalty_label = self.get_smoothness_penalty(label)
        penalty_active_p = self.get_smoothness_penalty(labels_p)
        penalty_active_q = self.get_smoothness_penalty(labels_q)

        indices_p, indices_q = self.neighbors
        is_p_in_range = indices_p[1, :] + labels_q < width
        is_q_in_range = indices_q[1, :] + labels_p < width

        num_neighbors = self.neighbors.T.shape[0]
        for neighbor_index in range(self.neighbors.shape[2]):
            indices_y, indices_x = self.neighbors.T[neighbor_index]
            node_l_p, node_l_q = self.nodes_label[indices_y, indices_x]
            if node_l_p != self.NODE_ABSENT and node_l_q != self.NODE_ABSENT:
                penalty = penalty_label[neighbor_index]
                if node_l_p != self.NODE_ALPHA and node_l_q != self.NODE_ALPHA:
                    self.add_smoothness_weights(g, node_l_p, node_l_q, 0, penalty, penalty, 0)
                elif node_l_p != self.NODE_ALPHA:
                    g.add_tedge(node_l_p, penalty, 0)
                elif node_l_q != self.NODE_ALPHA:
                    g.add_tedge(node_l_q, penalty, 0)

            node_a_p, node_a_q = self.nodes_active[indices_y, indices_x]
            is_p_active, is_q_active = self.is_node_active[indices_y, indices_x]
            label_p, label_q = labels_p[neighbor_index], labels_q[neighbor_index]
            penalty_p, penalty_q = penalty_active_p[neighbor_index], penalty_active_q[neighbor_index]

            if label_p == label_q:
                if not is_p_active or not is_q_active:
                    continue
                self.add_smoothness_weights(g, node_a_p, node_a_q, 0, penalty_p, penalty_p, 0)

            if is_p_active and is_q_in_range[neighbor_index]:
                g.add_tedge(node_a_p, penalty_p, 0)

            if is_q_active and is_p_in_range[neighbor_index]:
                g.add_tedge(node_a_q, penalty_q, 0)


    def get_smoothness_penalty(self, labels):
        _, width = self.image_shape
        indices_p, indices_q = self.neighbors

        indices_p_shifted = np.copy(indices_p[:, self.is_left_under])
        indices_q_shifted = np.copy(indices_q[:, self.is_left_under])
        is_oob_p = indices_p_shifted[1, :] >= width - labels
        indices_p_shifted[1, :] = np.clip(indices_p_shifted[1, :] + labels, 0, width - 1)
        is_oob_q = indices_q_shifted[1, :] >= width - labels
        indices_q_shifted[1, :] = np.clip(indices_q_shifted[1, :] + labels, 0, width - 1)
        diff_right = self.image_left[list(indices_p_shifted)] - self.image_right[list(indices_q_shifted)]

        smoothness = np.full(indices_p.shape[1], self.smoothness_cost_low, dtype=np.float)
        is_left_under = np.copy(self.is_left_under)
        is_left_under[is_left_under] = np.abs(diff_right) < self.smoothness_threshold
        smoothness[is_left_under] = self.smoothness_cost_high

        is_left_under = np.copy(self.is_left_under)
        is_left_under[is_left_under] = is_oob_p | is_oob_q
        smoothness[is_left_under] = 0

        return smoothness

    def add_smoothness_weights(self, g, node1, node2, w1, w2, w3, w4):
        w0 = w1 - w2
        g.add_tedge(node1, w4, w2)
        g.add_tedge(node2, 0, w0)
        g.add_edge(node1, node2, 0, w3 - w4 - w0)

    def add_uniqueness_term(self, g, labels, label):
        _, width = self.image_shape
        indices_y, indices_x = self.image_indices
        indices_shifted = indices_x + labels - label
        is_shift_valid = indices_shifted < width
        indices_shifted = np.clip(indices_shifted, 0, width - 1)
        forbid = self.is_node_active & is_shift_valid
        forbid_label = self.nodes_label[indices_y, indices_shifted][forbid]
        forbid_active = self.nodes_active[forbid]
        self.add_uniqueness_weights(g, forbid_active, forbid_label)

        is_node_label = self.nodes_label != self.NODE_ABSENT
        forbid = self.is_node_active & is_node_label
        self.add_uniqueness_weights(g, self.nodes_active[forbid], self.nodes_label[forbid])

    def add_uniqueness_weights(self, g, sources, targets):
        for i in range(sources.size):
            g.add_edge(sources[i], targets[i], sys.maxsize, 0)

    def update_labels(self, g, labels, label):
        is_node_active = np.copy(self.is_node_active)
        if is_node_active.any():
            nodes_active = self.nodes_active[is_node_active]
            is_node_active[is_node_active] = g.get_grid_segments(nodes_active)
            labels[is_node_active] = self.LABEL_OCCLUDED

        is_node_label = np.copy(self.is_node_label)
        if is_node_label.any():
            nodes_label = self.nodes_label[is_node_label]
            is_node_label[is_node_label] = g.get_grid_segments(nodes_label)
            labels[is_node_label] = label
