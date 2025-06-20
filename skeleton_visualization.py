from funcs.custom.skeleton_extraction import Skeletonize_IBP
from funcs.custom.skeleton_extraction_3d import Skeletonize_GBO_3D
from funcs.loss_function.clDice import SoftSkeletonize
import os
import cv2
import numpy as np
import torch
import time

num_iter = 5
skel_gbo = Skeletonize_IBP(num_iter=num_iter).cuda()
skel_gbo_3d = Skeletonize_GBO_3D(probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='EulerCharacteristic', num_iter=num_iter).cuda()
skel_cldice = SoftSkeletonize(num_iter=num_iter).cuda()

rv_dir = "assets/datasets/OCTA-500/RV"
save_dir = "skeleton"
os.makedirs(save_dir, exist_ok=True)

sample_id = 10001

total_time = 0

for sample_id in range(10001, 10501):
    label = cv2.imread("{}/{}.bmp".format(rv_dir, sample_id), cv2.IMREAD_GRAYSCALE)
    label = label[np.newaxis, np.newaxis, :] / 255
    label = torch.tensor(label).cuda().type(torch.cuda.FloatTensor)

    start = time.perf_counter()
    skel = skel_cldice(label)
    end = time.perf_counter()

    total_time += (end - start)

    skel = skel.cpu().numpy()[0][0] * 255
    cv2.imwrite("{}/cldice_{}_{}.png".format(save_dir, sample_id, num_iter), skel.astype(np.uint8))

print(f"used time: {total_time / 500} s")