import numpy as np
import torch
import imgaug.augmenters as iaa

def convert_image_to_tensor(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    else:
        if image.shape[2] < 10:
            image = np.transpose(image, (2, 0, 1))

    if image.max() > 1: image = image / 255
    return torch.tensor(image)

class ShuffleChannels(iaa.meta.Augmenter):
    def __init__(self, name=None, random_state=None):
        super().__init__(name=name, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return [image[..., random_state.permutation(image.shape[-1])] for image in images]

    def get_parameters(self):
        return []
    
class RandomChannelToThree(iaa.meta.Augmenter):
    def __init__(self, name=None, random_state=None):
        super().__init__(name=name, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        augmented_images = []
        for image in images:
            channel = random_state.randint(0, image.shape[-1])
            augmented_image = np.stack([image[..., channel]] * 3, axis=-1)
            augmented_images.append(augmented_image)
        return augmented_images

    def get_parameters(self):
        return []
    
seq = iaa.Sequential([
    iaa.Fliplr(0.1), # horizontal flips
    iaa.Flipud(0.1), # vertical flips
    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1))),
    iaa.Sometimes(0.1, iaa.LinearContrast((0.75, 1.5))),
    iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
    iaa.Sometimes(0.1, iaa.Rotate(rotate=(-10, 10), mode='constant')),
    iaa.Sometimes(0.1, iaa.Sharpen((0.0, 0.5))),
    iaa.Sometimes(0.1, iaa.ElasticTransformation(sigma=15)),
    iaa.Sometimes(0.1, iaa.ImpulseNoise(p=0.05)),
    iaa.Sometimes(0.1, iaa.Dropout([0.05, 0.2])),
    iaa.Sometimes(0.1, ShuffleChannels()),
    iaa.Sometimes(0.1, RandomChannelToThree()),
], random_order=True) # apply augmenters in random order

def data_augmentation(image, mask):
    to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
    is_image_2ch, is_mask_2ch = bool(len(image.shape) == 2), bool(len(mask.shape) == 2)
    if is_image_2ch: image = to_3ch(image)
    if is_mask_2ch: mask = to_3ch(mask)
    images, masks = np.expand_dims(image, axis=0), np.expand_dims(mask, axis=0)
    images_aug, masks_aug = seq(images=images, segmentation_maps=masks)
    images_aug, masks_aug = images_aug[0], masks_aug[0]
    if is_image_2ch: images_aug = images_aug[:,:,0]
    if is_mask_2ch: masks_aug = masks_aug[:,:,0]
    return images_aug, masks_aug
     