from utils import read_json, write_json
import os
import logging

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
data_dir = os.path.join(_root, "yolo_database")
annots = read_json(os.path.join(_root, "annot.json"))
classes = read_json(os.path.join(_root, "classes.json"))

for dr in os.listdir(data_dir):
    logging.warning(f"working on {data_dir}/{dr}")
    # target_annot = dict()
    # target_classes = dict()
    target_annot = read_json(f"{data_dir}/{dr}/annot.json")
    target_classes = read_json(f"{data_dir}/{dr}/classes.json")
    print(len(target_classes.keys())==len(os.listdir(f"{data_dir}/{dr}/images")))
    print(len(target_annot.keys())==len(os.listdir(f"{data_dir}/{dr}/images")))
    # for im_name in os.listdir(f"{data_dir}/{dr}/images"):
    #     target_annot[im_name] = annots[im_name][1:]
    #     target_classes[im_name] = classes[im_name][1:]
    # write_json(target_annot, f"{data_dir}/{dr}/annot.json")
    # write_json(target_classes, f"{data_dir}/{dr}/classes.json")
    # for k, v in target_annot.items():
    #     target_annot[k] = target_annot[k][1:]
    #     target_classes[k] = target_classes[k][1:]

    # write_json(__, f"{data_dir}/{dr}/classes.json")
    

