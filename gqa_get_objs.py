import h5py
import os
from tqdm import tqdm
import json
from collections import defaultdict
from utils import makeJson

#run 2
file2idx2img = defaultdict(dict)
img2bb = defaultdict(list)
img2n_objs = {}

info = json.load(open("gqa_data/gqa_objects_info.json", "r"))
h5s = [
        (
        "gqa_data/" + o,
        o.split("_")[-1].split(".")[0]
        ) for o in os.listdir("gqa_data/") if o[-2:] == "h5"
        ]

for img_id, d in tqdm(info.items()):
    file2idx2img[str(d["file"])][d["idx"]] = img_id
    img2n_objs[img_id] = d["objectsNum"]

for (h5, file_num) in tqdm(h5s):
    f = h5py.File(h5, 'r')
    objs = f["bboxes"]
    for idx, bb in enumerate(objs):
        img = file2idx2img[str(file_num)][idx]
        n = img2n_objs[img]
        boxes = bb.tolist()[:n]
        img2bb[img].append({
                "box": boxes,
                })

makeJson(img2bb, "custom_json/img2gqa_rcnn")
#makeJson(img2n_objs, "custom_json/img2n_objs")
#makeJson(img2areas, "custom_json/img2areas")
