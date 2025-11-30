import json

"""
what is the diff range we have? what is the distribution?
how is the std correlated to diff? if we choose diff > 2 to be wrong and get the std min as the threshold. how well can we predict all the diff > 2
gt, mu, std, diff
"""

with open("./results.json", "r") as f:
    file = json.load(f)

unacceptable_diff = set(k for k,v in file.items() if v[3]>5)
print("Incorrect model prediction percentage:", len(unacceptable_diff)/len(file))
min_std = file[min(unacceptable_diff, key=lambda x:file[x][2])][2]
# print("Min std is", min_std)

std_threshold = 2
std_filtered = set(k for k, v in file.items() if v[2] > std_threshold)
print("Percentage of predictions we are flagging as not sure:", len(std_filtered)/len(file))

correctly_guessed = len(unacceptable_diff.intersection(std_filtered))
left_over = unacceptable_diff - std_filtered
print("With human intervention,", (len(file)-len(left_over))/len(file), "percent of predictions have error of below 1.25mm")
precision = correctly_guessed/len(std_filtered)
recall = correctly_guessed/len(unacceptable_diff)
    
print("Precision", precision)
print("Recall", recall)
