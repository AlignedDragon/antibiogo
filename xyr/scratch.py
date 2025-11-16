import json
import os 
from utils import root_path, IMG_SIZE
import PIL


zones = os.path.join(root_path, "ihz_1024.json")
images = os.path.join(root_path, "complete")
patches = os.path.join(root_path, "patches")
radii = {}

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

# some need padding for 256 256
count = 0
for k, v_list in zones.items():
    img = padding(k, [1024 + 256, 1024 + 256])

    for i in range(len(v_list)):
        x,y,r,d = v_list[i][0] + 128, v_list[i][1] + 128, v_list[i][2], 128
        area = (x-d, y-d, x+d, y+d)
        img.crop(area).save(str(os.path.join(patches, f"{k.split('.')[0]}_{i}.jpg")), "JPEG")
        radii[f"{k.split('.')[0]}_{i}.jpg"] = r
    count +=1
    if count % 20 == 0:
        print(f"{count} - done")
    if count > 500:
        break

with open(os.path.join(root_path, "radii.json"), 'w') as f:
    json.dump(radii, f)



