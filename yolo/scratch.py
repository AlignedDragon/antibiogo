from dataloader import orig_train_batches
import json

# dim = {"im": [], "cl": [], "bx": []}
# # batch =  orig_train_batches.take(1)
# for i in orig_train_batches:
#     print(i[1]["boxes"].numpy()[1])
#     quit()
#     dim["im"].append(tuple(i[0].shape))
#     dim["cl"].append(tuple(i[1]['classes'].shape))
#     dim["bx"].append(tuple(i[1]['boxes'].shape))
# with open("./dims.json", 'w') as f:
#     json.dump(dim, f)

with open("./dims.json", 'r') as f:
    dims = json.load(f)
for k, v_list in dims.items():
    if k!="im":
        for v in v_list:
            if v[1]!=11:
                print("ishkal", k , v)