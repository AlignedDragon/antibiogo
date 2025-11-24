import json
import os 
from utils import root_path, IMG_SIZE
import PIL
from tqdm import tqdm

split = "val"
zones = os.path.join(*[root_path, "complete", "ihz_1024.json"])
images = os.path.join(*[root_path, f"yolo_database/{split}","images"])
patches = os.path.join(*[root_path, f"xyr_database/{split}","images"])
radii_path = os.path.join(*[root_path, f"xyr_database/{split}", "radii.json"])
radii = {}

# with open(radii_path, "r") as f:
#     print(len(json.load(f)))
#     print(len(os.listdir(patches)))

with open(zones) as f:
    zones = json.load(f)

def padding(img_name, expected_size):
    img = PIL.Image.open(os.path.join(images, img_name))
    desired_size = expected_size

    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]

    pad_width = delta_width // 2
    pad_height = delta_height // 2

    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return PIL.ImageOps.expand(img, padding)

for k in tqdm(os.listdir(images)):
    img = padding(k, [1024 + 256, 1024 + 256])

    v_list = zones[k]

    for i in range(len(v_list)):
        x,y,r,d = v_list[i][0] + 128, v_list[i][1] + 128, v_list[i][2], 128
        area = (x-d, y-d, x+d, y+d)
        img.crop(area).save(str(os.path.join(patches, f"{k.split('.')[0]}_{i}.jpg")), "JPEG")
        radii[f"{k.split('.')[0]}_{i}.jpg"] = r


with open(radii_path, 'w') as f:
    json.dump(radii, f)



