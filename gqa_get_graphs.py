from collections import defaultdict, Counter
from utils import loadJson, readFile, makeTxt, makeJson, loadGraphs
from tqdm import tqdm

#1 run first
#get all
#attributes
#relations
#objects

#things we want to collect as sets
relations = set()
attributes = set()
objects = set()

#things we want to collect as dictionaries
obj2relType = defaultdict(Counter)
id2ids = defaultdict(list)
imgId2objs = defaultdict(dict)
img2info = defaultdict(dict)
imgId2bb = defaultdict(list)
obj2id = defaultdict(list)
id2obj = {}
obj2attr = defaultdict(Counter)

#load data
graphs = loadGraphs("train", "val")

for img_id, scene_i in tqdm(graphs.items()):
    for obj_id, obj in scene_i["objects"].items():
        img2info[img_id] = {"relations": [], "attributes": []}
        obj_name = obj["name"]
        objects.add(obj_name)

        id2obj[obj_id] = obj_name
        obj2id[obj_name].append(obj_id)

        imgId2objs[img_id][obj_id] = obj_name
        imgId2bb[img_id].append({"box": [obj["x"], obj["y"], obj["w"], obj["h"]], "name": obj_name})

        #make attributes info
        for att in obj["attributes"]:
            attributes.add(att)
            obj2attr[obj_name][att] += 1
            if att not in img2info[img_id]["attributes"]:
                img2info[img_id]["attributes"].append(att)

        for related_info in obj["relations"]:
            relation = related_info["name"]
            relations.add(relation)
            obj2relType[obj_name][relation] += 1
            id2ids[obj_id].append(related_info["object"])
            if relation not in img2info[img_id]["relations"]:
                img2info[img_id]["relations"].append(relation)

#things we want to collect as dictionaries
obj2related = defaultdict(Counter)
for img_id, scene_i in tqdm(graphs.items()):
    for obj_id, obj in scene_i["objects"].items():
        obj_name = obj["name"]
        for related_info in obj["relations"]:
            obj_related = related_info["object"]
            obj2related[obj_name][id2obj[obj_related]] += 1

makeJson(obj2related, "custom_json/obj2related")
makeTxt(attributes, "custom_txt/attributes")
makeTxt(objects, "custom_txt/objects")
makeTxt(relations, "custom_txt/relations")
makeJson(imgId2objs, "custom_json/img2objs")
makeJson(imgId2bb, "custom_json/img2sg_bb")
makeJson(img2info, "custom_json/img2info")
makeJson(obj2attr, "custom_json/obj2attr")
makeJson(obj2id, "custom_json/obj2id")
makeJson(id2obj, "custom_json/id2obj")
makeJson(obj2relType, "custom_json/obj2relType")
makeJson(id2ids, "custom_json/id2ids")
