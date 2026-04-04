import json 
import logging

def read_json(path):
    if type(path)!= str or str=="":
        logging.warning("invalid path variable")
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data 

def write_json(data, path):
    logging.warning(f"Writing JSON to {path}")
    with open(path, "w") as f:
        json.dump(data, f)
    logging.warning(f"Done writing JSON to {path}")
    