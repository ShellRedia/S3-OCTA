from funcs.dataset.octa_500 import OCTA_500_Dataset, OCTA_500_SAM_Dataset
from funcs.dataset.rose_o import ROSE_O_Dataset
from funcs.dataset.octa_25k import OCTA_25K_Dataset

from torch.utils.data import ConcatDataset

def get_datasets(dataset_name, label_type):
    if "OCTA-500" in dataset_name:
        dataset_class = OCTA_500_SAM_Dataset if "SAM" in dataset_name else OCTA_500_Dataset
        if "3M" in dataset_name and "6M" in dataset_name:
            fov = "3M_6M"
        else:
            fov = "3M" if "3M" in dataset_name else "6M"
        get_octa_dataset = lambda subset: dataset_class(label_type=label_type, fov=fov, subset=subset)
        dataset_train, dataset_val, dataset_test = map(get_octa_dataset, ["train", "val", "test"])
    
    if "ROSE" in dataset_name:
        get_octa_dataset = lambda subset: ROSE_O_Dataset(label_type=label_type, subset=subset)
        dataset_train, dataset_val, dataset_test = map(get_octa_dataset, ["train", "val", "test"])

    if "OCTA-25K" in dataset_name:
        if "3M" in dataset_name and "6M" in dataset_name:
            fov = "3M_6M"
        else:
            fov = "3M" if "3M" in dataset_name else "6M"
        get_octa_dataset = lambda subset: OCTA_25K_Dataset(label_type=label_type, fov=fov, subset=subset)
        dataset_train, dataset_val, dataset_test = map(get_octa_dataset, ["train", "val", "test"])

    if dataset_name == "Merge":
        get_octa_dataset = lambda subset: OCTA_500_Dataset(label_type=label_type, fov="3M_6M", subset=subset)
        dataset_train_1, dataset_val_1, dataset_test_1 = map(get_octa_dataset, ["train", "val", "test"])
        get_octa_dataset = lambda subset: ROSE_O_Dataset(label_type=label_type, subset=subset)
        dataset_train_2, dataset_val_2, dataset_test_2 = map(get_octa_dataset, ["train", "val", "test"])

        dataset_train = ConcatDataset([dataset_train_1, dataset_train_2])
        dataset_val = ConcatDataset([dataset_val_1, dataset_val_2])
        dataset_test = ConcatDataset([dataset_test_1, dataset_test_2])
        
    return dataset_train, dataset_val, dataset_test
