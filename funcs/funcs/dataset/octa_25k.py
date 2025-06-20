from torch.utils.data import Dataset
from funcs.custom.image_process import convert_image_to_tensor, data_augmentation
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

class OCTA_25K_Dataset(Dataset):
    def __init__(
            self, 
            data_dir="assets/datasets/OCTA-25K",
            label_type="RV",
            fov="3M",
            subset="train",
            preload=True
        ):
        self.data_dir, self.label_type, self.fov, self.subset, self.preload = data_dir, label_type, fov, subset, preload
        self.augmenter = iaa.Resize({"height": 512, "width": 512})

        if fov == "3M": 
            sample_ids = list({"train":range(1, 21), "val":range(1, 21), "test":range(1, 101)}[subset])
        elif fov == "6M": 
            sample_ids = list({"train":range(101, 121), "val":range(101, 121), "test":range(101, 201)}[subset])
        else:
            sample_ids = list({"train":range(1, 21), "val":range(1, 21), "test":range(1, 101)}[subset]) \
                        + list({"train":range(101, 121), "val":range(101, 121), "test":range(101, 201)}[subset])
        
        self.code2quality = {1:"ungradable", 2:"gradable", 3:"outstanding"}

        self.offset_factor = 10 ** 5

        self.sample_ids = []
        for sample_id in sample_ids:
            for image_quality in range(1,4):
                self.sample_ids.append(image_quality * self.offset_factor + sample_id)

        self.sample_dct = {}

        if preload:
            for sample_id in self.sample_ids:
                self.sample_dct[sample_id] = self.get_sample_by_id(sample_id)


        
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
        image_quality, sample_id = sample_id // self.offset_factor, sample_id % self.offset_factor

        image = cv2.imread("{}/{}/{}/{:0>3}.png".format(self.data_dir, self.fov, self.code2quality[image_quality], sample_id), cv2.IMREAD_GRAYSCALE)
        mask = np.random.binomial(1, 0.5, size=image.shape)
        image = np.stack([image, image, image], axis=0)
        
        return image, mask
