from utils import makeJson, loadJson, readFile
from nltk.corpus import wordnet as wn

r_o =set()
r_r =set()
r_a =set()
vg_objs = readFile("custom_txt/vg_objects")
vg_rels = readFile("custom_txt/vg_relations")
vg_attrs = readFile("custom_txt/vg_attributes")
objs = readFile("custom_txt/objects")
rels = readFile("custom_txt/relations")
attrs = readFile("custom_txt/attributes")
osyn = loadJson("custom_json/obj_synonyms")
rsyn = loadJson("custom_json/rel_synonyms")
#vg_rsyn = readFile("custom_json/vg_rel_synonyms")
#vg_osyn = readFile("custom_json/vg_obj_synonyms")

osynf = set()
rsynf = set()

for o in osyn.values():
    for s in o:
        osynf.add(s)

for o in rsyn.values():
    for s in o:
        rsynf.add(s)

for o in vg_objs:
    if len(o.split(" ")) > 1:
        pass
    else:
        if o in objs or wn.morphy(o) in objs:
            pass
        elif o in osynf:
                pass
        else:
            r_o.add(o)

for o in vg_rels:
    if len(o.split(" ")) > 1:
        pass
    else:
        if o in rels or wn.morphy(o) in rels:
            pass
        elif o in rsynf:
            pass
        else:
            r_r.add(o)

for o in vg_attrs:
    if len(o.split("-")[-1].split(" ")) > 1:
        pass
    else:
        o = o.split("-")[-1]
        if o:
            o = o.lower()
            if o in attrs or wn.morphy(o) in attrs:
                pass
            else:
                r_a.add(o)
        elif o is not None and o.isdigit():
            r_a.add(o)

print("attrs",list( r_a)[:10], len(r_a))
print("objs", list(r_o)[:10], len(r_o))
print("rels", list(r_r)[:10], len(r_r))
