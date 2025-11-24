import json
import os, random
from tqdm import tqdm

root = "/users/msayfiddinov/scratch/antibiogo/"
db = "/users/msayfiddinov/scratch/antibiogo/yolo_database/"
train = os.path.join(db, "train")
val = os.path.join(db, "val")
test = os.path.join(db, "test")

# with open(root+"annot.json", "r") as f:
#     base_annot = json.load(f)  
# with open(root+"classes.json", "r") as f:
#     base_classes = json.load(f)

# for fo in os.listdir(db):
#     folder = os.path.join(*[db,fo, "images"])
#     annot = {}
#     classes = {}
#     for file in os.listdir(folder):
#         annot[file] = base_annot[file]
#         classes[file] = base_classes[file]
#     with open(db+f"{fo}/annot.json", "w") as f:
#         json.dump(annot, f)
#     with open(db+f"{fo}/classes.json", 'w') as f:
#         json.dump(classes, f)
        

with open(val+"/annot.json", "r") as f:
    print(len(json.load(f)))
    print(len(os.listdir(val+"/images")))