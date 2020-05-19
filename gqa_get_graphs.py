"""
    info in form
    {
    img_id:
        [
            {
                "object": NAME,
                "id": ID,
                "box": [x,y,h,w],
                "attributes": [NAMES],
                "relations": [[RELATION, RELATED_ID]],
            }
        ]
    }
"""

from collections import defaultdict
from utils import loadJson, loadTxt, makeTxt, makeJson, loadGQAGraphs
from tqdm import tqdm


#data to make
obj2id = defaultdict(list)
id2obj = {}
graphs = {}
uniq_relations =  set()
uniq_attributes = set()
uniq_objects = set()

#load data
in_graphs = loadGQAGraphs("data_gqa", "train", "val")

# first pass to get obj2id and id2obj
for img_id, scene_i in tqdm(in_graphs.items()):
    for obj_id, obj in scene_i["objects"].items():
        obj_name = obj["name"]
        uniq_objects.add(obj_name)
        id2obj[obj_id] = obj_name
        obj2id[obj_name].append(obj_id)
        for att in obj["attributes"]:
            uniq_attributes.add(att)
        for related_info in obj["relations"]:
            relation = related_info["name"]
            uniq_relations.add(relation)

#secnd pass to construct/reformat graphs
for img_id, scene_i in tqdm(in_graphs.items()):
    img_info = []
    for obj_id, obj in scene_i["objects"].items():
        rels = obj["relations"]
        img_info.append({
            "name": obj["name"],
            "id": obj_id,
            "box": [obj["x"], obj["y"], obj["w"], obj["h"]],
            "attributes": obj["attributes"],
            "relations": [[r_id["name"], r_id["object"]]\
                for r_id in rels]
        })
    graphs[img_id] = img_info

makeTxt(uniq_attributes, "phase_1/attributes")
makeTxt(uniq_objects, "phase_1/objects")
makeTxt(uniq_relations, "phase_1/relations")
makeJson(graphs, "phase_1/gqa_img2info")
makeJson(obj2id, "phase_1/gqa_obj2id")
makeJson(id2obj, "phase_1/gqa_id2obj")
