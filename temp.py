import os

dataset_dir = "assets/datasets/OCTA-25K/3M"

for sub_dir in "gradable", "outstanding", "ungradable":
    image_dir = "{}/{}".format(dataset_dir, sub_dir)
    for image_file in os.listdir(image_dir):
        sample_id = int(image_file[:-4]) + 1
        os.rename("{}/{}".format(image_dir, image_file), "{}/{:0>3}.png".format(image_dir, sample_id))