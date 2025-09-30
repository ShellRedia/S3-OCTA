import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

class SoftSkeletonize_clDice:
    def __init__(self, num_iter=40):
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 2:  # 2D case (batch, channel, height, width)
            p1 = -maximum_filter(-img, size=(3, 1), mode='constant')
            p2 = -maximum_filter(-img, size=(1, 3), mode='constant')
            return np.minimum(p1, p2)
        elif len(img.shape) == 3:  # 3D case (batch, channel, depth, height, width)
            p1 = -maximum_filter(-img, size=(3, 1, 1), mode='constant')
            p2 = -maximum_filter(-img, size=(1, 3, 1), mode='constant')
            p3 = -maximum_filter(-img, size=(1, 1, 3), mode='constant')
            return np.minimum(np.minimum(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 2:  # 2D case
            return maximum_filter(img, size=(3, 3), mode='constant')
        elif len(img.shape) == 3:  # 3D case
            return maximum_filter(img, size=(3, 3, 3), mode='constant')

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        if img.max() > 1: img = img / 255
        img1 = self.soft_open(img)
        skel = np.maximum(img - img1, 0)

        for i in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = np.maximum(img - img1, 0)
            skel = skel + np.maximum(delta - skel * delta, 0)
            cv2.imwrite("skel_{:0>2}.png".format(i), skel * 255)
        return img
    
class SkeletonExtraction:
    def __init__(self):
        self.soft_skeletonize_clDice = SoftSkeletonize_clDice(num_iter=10)

    def get_skeleton(self, binary_mask):
        return self.soft_skeletonize_clDice.soft_skel(binary_mask)

class Skeletonize_IBP(torch.nn.Module):
    def __init__(self, num_iter=5):
        super(Skeletonize_IBP, self).__init__()
        self.num_iter = num_iter

    def forward(self, img):
        img = img.squeeze(1)
        img = F.pad(img, (1, 1, 1, 1), value=0)

        img = self._stochastic_discretization(img)

        for current_iter in range(self.num_iter):

            x_offsets = [0, 1, 0, 1]
            y_offsets = [0, 0, 1, 1]
            
            is_endpoint = self._single_neighbor_check(img)

            for x_offset, y_offset in zip(x_offsets, y_offsets):

            # At each sub-iteration detect all simple points and delete all simple points that are not end-points
                is_simple = self._euler_characteristic_simple_check(img[:, x_offset:, y_offset:])

                deletion_candidates = is_simple * (1 - is_endpoint[:, x_offset::2, y_offset::2])
                img[:, x_offset::2, y_offset::2] = torch.min(img[:, x_offset::2, y_offset::2].clone(), 1 - deletion_candidates)


        img = img[:, 1:-1, 1:-1]

        return img.unsqueeze(1)


    def _stochastic_discretization(self, img, beta=0.33, tau=1.0):
        """
        Function to binarize the image so that it can be processed by our skeletonization method.
        In order to remain compatible with backpropagation we utilize the reparameterization trick and a straight-through estimator.
        """

        alpha = (img + 1e-8) / (1.0 - img + 1e-8)

        uniform_noise = torch.rand_like(img)
        uniform_noise = torch.empty_like(img).uniform_(1e-8, 1 - 1e-8)
        logistic_noise = (torch.log(uniform_noise) - torch.log(1 - uniform_noise))

        img = torch.sigmoid((torch.log(alpha) + logistic_noise * beta) / tau)
        img = (img.detach() > 0.5).float() - img.detach() + img

        return img


    def _single_neighbor_check(self, img):
        """
        Function that characterizes points as endpoints if they have a single neighbor or no neighbor at all.
        """
        img = F.pad(img, (1, 1, 1, 1))

        bs = img.shape[0]

        # Check that number of ones in twentysix-neighborhood is exactly 0 or 1
        K = torch.tensor([[1.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]] * bs, device=img.device).view(1, bs, 3, 3)


        num_eight_neighbors = F.conv2d(img, K)
        condition1 = F.hardtanh(-(num_eight_neighbors - 2), min_val=0, max_val=1) # 1 or fewer neigbors
        
        return condition1

    # Specifically designed to be used with the eight-subfield iterative scheme from above.
    def _euler_characteristic_simple_check(self, img):
        """
        Function that identifies simple points by assessing whether the Euler characteristic changes when deleting it [1].
        In order to calculate the Euler characteristic, the amount of vertices, edges, faces and octants are counted using convolutions with pre-defined kernels.
        The function is meant to be used in combination with the subfield-based iterative scheme employed in the forward function.

        [1] Steven Lobregt et al. Three-dimensional skeletonization:principle and algorithm.
            IEEE Transactions on pattern analysis and machine intelligence, 2(1):75-77, 1980.
        """

        img = F.pad(img, (1, 1, 1, 1), value=0)
        
        bs = img.shape[0]

        mask = torch.ones_like(img)
        mask[:, 1::2, 1::2] = 0
        masked_img = img.clone() * mask

        # Count vertices
        vertices = F.relu(-(2.0 * img - 1.0))
        num_vertices = F.avg_pool2d(vertices, (3, 3), stride=2) * 9

        masked_vertices = F.relu(-(2.0 * masked_img - 1.0))
        num_masked_vertices = F.avg_pool2d(masked_vertices, (3, 3), stride=2) * 9

        # Count edges
        K_x_edge = torch.tensor([0.5, 0.5] * bs, device=img.device).view(1, bs, 2, 1)
        K_y_edge = torch.tensor([0.5, 0.5] * bs, device=img.device).view(1, bs, 1, 2)

        x_edges = F.relu(F.conv2d(-(2.0 * img - 1.0), K_x_edge))
        num_x_edges = F.avg_pool2d(x_edges, (2, 3), stride=2) * 6
        y_edges = F.relu(F.conv2d(-(2.0 * img - 1.0), K_y_edge))
        num_y_edges = F.avg_pool2d(y_edges, (3, 2), stride=2) * 6


        num_edges = num_x_edges + num_y_edges

        masked_x_edges = F.relu(F.conv2d(-(2.0 * masked_img - 1.0), K_x_edge))
        num_masked_x_edges = F.avg_pool2d(masked_x_edges, (2, 3), stride=2) * 6
        masked_y_edges = F.relu(F.conv2d(-(2.0 * masked_img - 1.0), K_y_edge))
        num_masked_y_edges = F.avg_pool2d(masked_y_edges, (3, 2), stride=2) * 6

        num_masked_edges = num_masked_x_edges + num_masked_y_edges

        # Count faces
        K_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]] * bs, device=img.device).view(1, bs, 2, 2)
        
        x_faces = F.relu(F.conv2d(-(2.0 * img - 1.0), K_face) - 0.5) * 2
        num_faces = F.avg_pool2d(x_faces, (2, 2), stride=2) * 4
        
        masked_x_faces = F.relu(F.conv2d(-(2.0 * masked_img - 1.0), K_face) - 0.5) * 2
        num_masked_faces = F.avg_pool2d(masked_x_faces, (2, 2), stride=2) * 4

        # Combined number of vertices, edges, faces and octants to calculate the euler characteristic
        euler_characteristic = num_vertices - num_edges + num_faces
        masked_euler_characteristic = num_masked_vertices - num_masked_edges + num_masked_faces

        # If the Euler characteristic is unchanged after switching a point from 1 to 0 this indicates that the point is simple
        euler_change = F.hardtanh(torch.abs(masked_euler_characteristic - euler_characteristic), min_val=0, max_val=1)
        is_simple = 1 - euler_change
        is_simple = (is_simple.detach() > 0.5).float() - is_simple.detach() + is_simple

        return is_simple