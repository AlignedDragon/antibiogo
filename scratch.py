import json


# if length < 17 -> catenat [0,0,0,0] so for the classes

with open("/users/msayfiddinov/scratch/antibiogo_data/annot.json", 'r') as f:
    annot = json.load(f)
with open("/users/msayfiddinov/scratch/antibiogo_data/classes.json", 'r') as f:
    classes = json.load(f)

for k, v_list in annot.items():
    if len(v_list) < 17:
        for i in range(17 - len(v_list)):
            annot[k].append([0.0,0.0,0.0,0.0])
            classes[k].append(0)

    if len(annot[k]) != len(classes[k]) or len(annot[k])!=17:
        print("ishkal ", k)
print("tamam")

with open("./annot.json", 'w') as f:
    json.dump(annot, f)
with open("./classes.json", 'w') as f:
    json.dump(classes, f)
