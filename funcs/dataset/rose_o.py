from torch.utils.data import Dataset
from funcs.custom.image_process import convert_image_to_tensor, data_augmentation
import os
import cv2
import numpy as np

class ROSE_O_Dataset(Dataset):
    def __init__(
            self, 
            data_dir="assets/datasets/ROSE-O",
            label_type="RV",
            subset="train"
        ):
        self.data_dir, self.label_type, self.subset = data_dir, label_type, subset

        self.sample_ids = list({"train":range(1, 31), "val":range(31, 40), "test":range(31, 40)}[subset])
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        image_ivc = cv2.imread("{}/img/IVC/{:0>2}.png".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_dvc = cv2.imread("{}/img/DVC/{:0>2}.tif".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_svc = cv2.imread("{}/img/SVC/{:0>2}.tif".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)

        image = np.stack([image_svc, image_dvc, image_ivc], axis=0)
        mask = cv2.imread("{}/gt/RV/{:0>2}.png".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)

        image =  np.transpose(image, (1, 2, 0))
        image, mask = map(lambda x: cv2.resize(x, (512, 512)), [image, mask])

        if bool(self.subset == "train"):
            image, mask = data_augmentation(image, mask)

        image, mask = map(convert_image_to_tensor, (image, mask))

        return image, mask, sample_id
