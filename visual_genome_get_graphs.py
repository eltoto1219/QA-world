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
import json
from utils import loadJson, loadTxt, makeTxt, makeJson, word2alternatives
from collections import defaultdict, Counter
from tqdm import tqdm
from autocorrect import Speller
SPELL = Speller(lang='en')
PUNC = set(w for w in "!@#$%^&*()-.?)")
TINY = False
MIN = 15
#data to load
relations = loadTxt("phase_1/relations")
attributes = loadTxt("phase_1/attributes")
objects = loadTxt("phase_1/objects")
#answers = loadTxt("phase_1/gqa_answers")
#data to make
obj2id = defaultdict(list)
id2obj = {}
uniq_relations =  Counter()
uniq_attributes = Counter()
uniq_objects = Counter()
graphs = {}
obj_synonyms = defaultdict(list)
rel_synonyms = defaultdict(list)

#functions
def getObjfromID(obj_id, img_info):
    idx = False
    for i, x in enumerate(img_info):
        if x["id"] == obj_id:
            idx = i
            break
    if idx:
        return img_info.pop(idx)
    else:
        pass

def customCheck(word, set2check):
    og = word
    if not word or len(word) == 1:
        return []
    chars = [w for w in word]
    if list(filter(lambda x: x.isdigit(), chars)):
        return []
    bad = set(chars).intersection(PUNC)
    for b in bad:
        word = word.replace(b, "")
    try:
        spelling = SPELL(word)
        if type(spelling) is list:
            spelling = spelling[0]
        assert type(spelling) == str
    except Exception:
        spelling = "INVALID"
    word = spelling if word!= spelling else word
    if len([c for c in chars if c == " "]) == 1:
        split = word.split(" ")
        if (split[0] in attributes \
                or split[0].isdigit()\
                or len(split[0]) ==1)\
                or split[0] in relations\
                and split[1] in set2check:
                    word = split[1]
    cands = word2alternatives(word)
    if og not in cands:
        cands += og
    return cands

print("LOADING")
vg = loadJson("data_visual_genome/scene_graphs")
for scene in tqdm(vg):

    img_id = scene["image_id"]
    objs = scene["objects"]
    relationships = scene["relationships"]
    img_info = []

    #GET OBJECTS
    for o in objs:
        #get attributes
        try:
            attrs = o["attributes"]
            attrs = None
            fil_attrs = []
            if attrs is not None:
                for a in attrs:
                    a_set = set(word2alternatives(a)).\
                            intersections(attributes)
                    if not a_set:
                        uniq_attributes[a] += 1
                    fil_attrs.append(next(iter(a)))
        except:
            fil_attrs = []
        #get names
        all_names = set()
        names = o["names"]+[s.split(".")[0].replace("_"," ")\
                for s in o["synsets"]]
        names = list(map(lambda x: x.lower(), names))
        for n in names:
            for n_alt in customCheck(n, objects):
                all_names.add(n_alt)
        # the name must exist
        if all_names:
            repr_name = all_names.intersection(objects)
            all_names = list(all_names)
            if repr_name:
                repr_name = next(iter(repr_name))
                all_names.remove(repr_name)
            else:
                repr_name = all_names.pop(0)
                uniq_objects[repr_name] += 1

            #make object entry
            img_info.append({
                    "id": o["object_id"],
                    "object": repr_name,
                    "box": [o["x"], o["y"], o["w"], o["h"]],
                    "attributes": fil_attrs,
                    "relations": []
                    })

            #fill in other data
            obj2id[repr_name].append(o["object_id"])
            id2obj[o["object_id"]] = repr_name
            obj_synonyms[repr_name] += [na for na in all_names\
                    if na not in obj_synonyms[repr_name]]

    #GET RELATIONS
    for r in relationships:
        # get rels
        all_names = set()
        rels =  set([s.split(".")[0].replace("_", " ").lower()\
                for s in r["synsets"]] +  [r["predicate"].lower()])
        for n in rels:
            for n_alt in customCheck(n, relations):
                all_names.add(n_alt)
        if all_names and r["subject_id"] in id2obj and r["object_id"] in obj2id:
            repr_name = all_names.intersection(relations)
            all_names = list(all_names)
            if repr_name:
                repr_name = next(iter(repr_name))
                all_names.remove(repr_name)
            else:
                repr_name = all_names.pop(0)
                uniq_relations[repr_name] += 1

            # APPEND RELATION
            obj_data = getObjfromID(r["object_id"], img_info)
            obj_data["relations"].append([repr_name, r["subject_id"]])

            #insert back into entry
            img_info.append(obj_data)

            #update other data
            rel_synonyms[repr_name] += [r for r in all_names\
                    if r not in rel_synonyms[repr_name]]

    graphs[img_id] = img_info

# FILTER OUT ENTITIES BELOW THRESHHOLD
uniq_relations = {c for c,i in uniq_relations.items() if i >= MIN}
uniq_objects = {c for c,i in uniq_objects.items() if i >= MIN}
uniq_attributes = {c for c,i in uniq_attributes.items() if i >= MIN}
accept_rels = uniq_relations.union(relations)
accept_objs = uniq_objects.union(objects)
accept_atts = uniq_attributes.union(attributes)

print("FILTERING")
for k in tqdm(graphs):
    img_info = graphs[k]
    cleaned_info = []
    for obj_info in img_info:
        if obj_info["object"] in accept_objs:
            cleaned = {}
            cleaned["object"] = obj_info["object"]
            cleaned["id"] = obj_info["id"]
            cleaned["box"] = obj_info["box"]
            cleaned["attributes"] = [a for a in obj_info["attributes"]\
                    if a in accept_atts]
            cleaned["relations"] = [e for e in obj_info["relations"]\
                if e[0] in accept_rels and\
                e[1] in id2obj and id2obj[e[1]] in accept_objs]
            cleaned_info.append(cleaned)
    graphs[k] = cleaned_info

print("UNIQUE ATTRS", len(uniq_attributes))
print("UNIQUE OBJECTS", len(uniq_objects))
print("UNIQUE RELATIONS", len(uniq_relations))
makeTxt(uniq_attributes, "phase_2/visual_genome_attributes")
makeTxt(uniq_objects, "phase_2/visual_genome_objects")
makeTxt(uniq_relations, "phase_2/visual_genome_relations")
makeJson(graphs, "phase_2/visual_genome_graphs")
makeJson(obj2id, "phase_2/visual_genome_obj2id")
makeJson(id2obj, "phase_2/visual_genome_id2obj")
makeJson(rel_synonyms, "phase_2/visual_genome_rel_synonyms")
makeJson(obj_synonyms, "phase_2/visual_genome_obj_synonyms")
