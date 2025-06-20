import json
import os

# # 读取 JSON 文件
# with open('file.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

result_dir = "results/segmentation/train"

content_lst = []
for train_record in os.listdir(result_dir):
    content_lst.append(train_record)
    train_record_dir = "{}/{}".format(result_dir, train_record)
    for sub_item in os.listdir(train_record_dir):
        if "0200" == sub_item:
            content_lst.append("Finish")
        if sub_item.endswith(".json"):
            content_lst.append(sub_item)
            with open("{}/{}".format(train_record_dir, sub_item), 'r', encoding='utf-8') as file:
                content_lst.append(json.load(file))

with open('result_overview.txt', 'w', encoding='utf-8') as file:
    for line in content_lst:
        file.write(line + '\n')