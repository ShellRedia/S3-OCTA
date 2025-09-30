from torch.utils.data import Dataset
from funcs.custom.image_process import convert_image_to_tensor, data_augmentation
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

class OCTA_500_Dataset(Dataset):
    def __init__(
            self, 
            data_dir="assets/datasets/OCTA-500",
            label_type="RV",
            fov="3M",
            subset="train",
            preload=True
        ):
        self.data_dir, self.label_type, self.subset, self.preload = data_dir, label_type, subset, preload

        self.augmenter = iaa.Resize({"height": 512, "width": 512})

        if fov == "3M": 
            sample_ids = list({"train":range(10301, 10441), "val":range(10441, 10451), "test":range(10451, 10501)}[subset])
        elif fov == "6M": 
            sample_ids = list({"train":range(10001, 10181), "val":range(10181, 10201), "test":range(10201, 10301)}[subset])
        else:
            sample_ids = list({"train":range(10301, 10441), "val":range(10441, 10451), "test":range(10451, 10501)}[subset]) \
                        + list({"train":range(10001, 10181), "val":range(10181, 10201), "test":range(10201, 10301)}[subset])
        
        self.sample_dct = {}
        if preload:
            for sample_id in sample_ids:
                self.sample_dct[sample_id] = self.get_sample_by_id(sample_id)

        self.sample_ids = sample_ids
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        if self.preload:
            image, mask = self.sample_dct[sample_id]
        else:
            image, mask = self.get_sample_by_id(sample_id)


        image =  np.transpose(image, (1, 2, 0))
        
        # mask = np.expand_dims(mask, axis=-1)
        # image, mask = self.augmenter(images=[image], segmentation_maps=[mask])
        # image, mask = image[0], mask[0][:,:,0] 

        if bool(self.subset == "train"):
            image, mask = data_augmentation(image, mask)

        image, mask = map(convert_image_to_tensor, (image, mask))

        return image, mask, sample_id

    def get_sample_by_id(self, sample_id):
        image_full = cv2.imread("{}/ProjectionMaps/OCTA(FULL)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_ilm_opl = cv2.imread("{}/ProjectionMaps/OCTA(ILM_OPL)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_opl_bm = cv2.imread("{}/ProjectionMaps/OCTA(OPL_BM)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)

        image = np.stack([image_full, image_ilm_opl, image_opl_bm], axis=0)
        mask = cv2.imread("{}/{}/{}.bmp".format(self.data_dir, self.label_type, sample_id), cv2.IMREAD_GRAYSCALE)

        return image, mask
    


class OCTA_500_SAM_Dataset(Dataset):
    def __init__(
            self, 
            data_dir=r"D:\PythonProjects\ImagePytorch\assets\datasets\OCTA-500",
            label_type="RV",
            fov="3M",
            subset="train"
        ):
        self.data_dir, self.label_type, self.subset = data_dir, label_type, subset

        if fov == "3M": 
            sample_ids = list({"train":range(10301, 10441), "val":range(10441, 10451), "test":range(10451, 10501)}[subset])
        elif fov == "6M": 
            sample_ids = list({"train":range(10001, 10181), "val":range(10181, 10201), "test":range(10201, 10301)}[subset])
        else:
            sample_ids = list({"train":range(10301, 10441), "val":range(10441, 10451), "test":range(10451, 10501)}[subset]) \
                        + list({"train":range(10001, 10181), "val":range(10181, 10201), "test":range(10201, 10301)}[subset])

        self.sample_ids = sample_ids
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        is_training = bool(self.subset == "train")

        sample_id = self.sample_ids[index]

        image_full = cv2.imread("{}/ProjectionMaps/OCTA(FULL)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_ilm_opl = cv2.imread("{}/ProjectionMaps/OCTA(ILM_OPL)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        image_opl_bm = cv2.imread("{}/ProjectionMaps/OCTA(OPL_BM)/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)

        image = np.stack([image_full, image_ilm_opl, image_opl_bm], axis=0)
        mask = cv2.imread("{}/RV/{}.bmp".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)

        image, mask = map(convert_image_to_tensor, (image, mask))
        return image, mask, sample_id