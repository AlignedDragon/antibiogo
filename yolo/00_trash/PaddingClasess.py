import numpy as np
from utils import classes_path
import json

dicty = json.load(open(classes_path))
nn = {}
for key, value in dicty.items():
    new = np.array(value, int) + 1
    nn[key] = [int(x) for x in np.pad(new,pad_width= (0, 16 - len(value)), constant_values= 0)]

with open('/home/muhammad-ali/working/detector/new.json', 'w') as f:
    json.dump(nn, f)