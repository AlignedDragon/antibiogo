import json

# get pellets, get 2 circular and 3 rectangular dishes. 
# -DishBoxes [x,y,w,h, IsCircular]
# -pellets [x,y,r]

# get annot -  image: annotation_list
# get classes -  image: class_labels

dish_path = "/home/muhammadali/datasets/anitbiogo_cleaned_data/DishBoxes_1024.json"
pellet_path = "/home/muhammadali/datasets/anitbiogo_cleaned_data/pellets_1024.json"

with open(dish_path) as f:
    dishes = json.load(f)
with open(pellet_path) as f:
    pellets = json.load(f)
for k, v in pellets.items():
    pellets[k] = [[v[i][0], v[i][1], v[i][2]*2, v[i][2]*2] for i in range(len(v))]

annot = {}
classes = {}
for k, v in dishes.items():
    annot[k] = [v[:4]]
    classes[k] = [2 if v[4]==True else 3]

for k, v in pellets.items():
    annot[k] += pellets[k]
    classes[k] += [1 for _ in range(len(pellets[k]))]

with open("./annot.json", "w") as f:
    json.dump(annot, f)
with open("./classes.json", "w") as f:
    json.dump(classes, f)
