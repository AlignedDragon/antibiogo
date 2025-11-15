import json
from os import path
import numpy as np

def trans(circles):
    for i in range(len(circles)):
        circles[i][2] *= 2
        circles[i].append(circles[i][2])
    return circles

root_path = '/home/muhammad-ali/working/detector'
img_pth = path.join(root_path,"base_data/ready")
annot_path = path.join(root_path, 'base_data/jsonFiles/circles_H1024.json')

annot = json.load(open(annot_path))
bbox = list(annot.values()).copy()
bbox = [trans(x) for x in bbox]
# print(annot['1.jpg'])
classes = {}
for key, values in annot.items():
    classes[key] = [0 for x in values]

with open(path.join(root_path, 'base_data/annot.json'), 'w') as f:
    json.dump(annot, f)
with open(path.join(root_path, 'base_data/classes.json'), 'w') as j:
    json.dump(classes, j)
print(len(annot), len(classes))