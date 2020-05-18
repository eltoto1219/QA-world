from collections import defaultdict, Counter
from functions import loadJson, readFile, makeTxt, makeJson, loadOGData
from tqdm import tqdm
import re
from shapely.geometry import Polygon
import statistics
from nltk.corpus import wordnet as wn #download('wordnet')`
from nltk import word_tokenize, pos_tag, download

#funcs
def posTag(sent):
    try:
        tokenized = word_tokenize(sent)
    except LookupError:
        download('punkt')
        tokenized = word_tokenize(sent)
    try:
        tagged = pos_tag(tokenized)
    except LookupError:
        download('averaged_perceptron_tagger')
    words = []
    for (word, pos) in tagged:
        if "N" in pos and word not in objects:
            words.append(word)
            UNKOWNS.add(word)
    return words

def calculate_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        denom = max(float(boxAArea + boxBArea - interArea), float(1))
        iou = interArea / denom
        # return the intersection over union value
        return iou * 100

def getInParens(s, d = None, word = True, u = None):
    if s.isdigit():
        return ACT
    else:
        r = longestInteger(s, len(s))
        if r.isdigit() and word:
            r = id2obj[r]
        if not r:
            r = s
        if r.isdigit() and word:
            r = id2obj[r]
        elif r.isdigit() and not word:
            pass
        elif not r.isdigit() and word:
            for o in objects:
                if " " + o + " " in " " + r + " ":
                    r = o
                    break
            if r in attributes:
                r = ATT
            elif r in relations:
                r = REL
        elif not r.isdigit() and not word and d is not None:
            for o in objects:
                if " " + o + " " in " " + r + " ":
                    r = o
                    break
            if r in attributes:
                r = ATT
            elif r in relations:
                r = REL
            elif r in objects:
                try:
                    r =d[r]
                except:
                    r = OOI
            elif r == "?":
                r = "[PAD]"
            else:
                u.add(r)
                r = UNK
        return str(r)

print("loading")
#things we want to collect as sets
relations = set(readFile("custom_txt/relations"))
attributes = set(readFile("custom_txt/attributes"))
objects = set(readFile("custom_txt/objects"))

#things we want to load as dicts
obj2related = loadJson("custom_json/obj2related")
imgId2objIds = loadJson("custom_json/imgId2objIds")
imgId2objs = loadJson("custom_json/imgId2objs")
imgId2coord  =  loadJson("custom_json/imgId2coord")
imgId2sizes  =  loadJson("custom_json/imgId2sizes")
obj2id = loadJson("custom_json/obj2id")
id2obj = loadJson("custom_json/id2obj")

splits = ["val","train"]

ATT = "[ATT]"
REL = "[REL]"
ACT = "[ACT]"
UNK = "[UNK]"
OOI = "[SUB]"

#things we want to create
UNKOWNS = set()
obj2child = defaultdict(list)
JUNK = {}

print("starting loop")
for split in splits:
    tosave = {}
    objId2attr = defaultdict(list)
    overlap_objs = {}

    print("split {}, subset {}".format(split, "balanced"))
    questions = loadOGData("balanced", split)
    print("loaded split")

    for qid, v in tqdm(questions.items()):
        try:
            longterm = True
            answer = v['answer']
            image = v["imageId"]
            objs = imgId2objs[image]
            ids = imgId2objIds[image]
            sizes = imgId2sizes[image]
            coords_alt = imgId2coord[image]
            bbs = [a + b for a,b, in zip(coords_alt, sizes)]
            o2bb = {o: bb for o, bb in zip(objs,bbs)}
            q_obj2id = {o: oid for o, oid in zip(objs, ids)}
            q_obj2id_save = [(o, oid) for o, oid in zip(objs, ids)]
            question = v['question']
            fa = v['fullAnswer'].replace(",", "")
            gt = {"rel": [], "att": [], "obj": []}
            detailed = v["types"]["detailed"]
            q2_obj2id = {id2obj[oid]: oid for oid in v["annotations"]["question"].values()}
            fl = [l["operation"]\
                for l in questions[qid]["semantic"]]

            obj_args = [getInParens(l["argument"], u = UNKOWNS)\
                for l in questions[qid]["semantic"]]

            id_args =  [getInParens(l["argument"],
                            word=False, d=q_obj2id, u = UNKOWNS)\
                for l in questions[qid]["semantic"]]


            layout = [l["operation"].split(" ")[0]\
                for l in questions[qid]["semantic"]]

        except Exception:
            JUNK[qid] = v["imageId"]
            print("junk")
            continue


        for o in objects:
            if o not in objs and (" " + o + " " in question or " " + o + "?" in question):
                if gt["obj"]:
                    obj2child[o].extend(gt["obj"])
                else:
                    obj2child[o].extend(gt["att"])
                cur = obj2child[o]
                obj2child[o] = list(set(cur))


        for a in attributes:
            if " " + a + " " in " " + fa or " " + a + "." in " " + fa and " not " not in question:
                gt["att"].append(a)
        for r in relations:
            if " " + r + " " in " " + fa or " " + r + "." in " " + fa:
                gt["rel"].append(r)
        for o in objects:
            if " " + o + " " in " " + fa or " " + o + "." in " " + fa:
                gt["obj"].append(o)


        #get overlap of objs
        if split == "val":
            bb_overlaps = []
            n_overlap = 0
            for q in q2_obj2id:
                if q in q_obj2id:
                    overlap_objs[q] = q2_obj2id[q]
                    n_overlap += 1
                    bb = o2bb[q]
                    max_overlap = 0
                    for bb2 in bbs:
                        if bb != bb2:
                            iou = calculate_iou(bb, bb2)
                            if iou > max_overlap:
                                max_overlap = iou
                                bb_save = bb2
                    bb_overlaps.append(max_overlap)
            try:
                bb_overlaps = statistics.mean(bb_overlaps)
            except:
                bb_overlaps = 0



            if n_overlap == 0 and gt["obj"]:
                overlap_objs[gt["obj"][0]] = None
                q2_obj2id[gt["obj"][0]] = None
                n_overlap += 1

            rels = []
            last_r = 0

            #get relations in question
            for r in relations:
                if " " + r + " " in question or " " + r + "?" in question:
                    idx = question.find(r)
                    last_r = max(idx, last_r)
                    rels.append((r, idx))
            rels = sorted(rels, key=lambda x: len(x[0]))

            filtered_rels = []
            for (r, idx) in rels:
                s = idx
                e = idx + len(r)
                for (r2, idx2) in rels:
                    if r2 == r or len(r2) >= len(r):
                        pass
                    else:
                        if s <= idx2 and e >= idx2:
                            pass
                        else:
                            filtered_rels.append(r)

            filtered_rels = list(set(filtered_rels))
            n_rels = len(filtered_rels)

            #see if longterm
            for o in objects:
                if o in question[last_r:]:
                    longterm = False


            if longterm:
                longterm = "longterm"
            else:
                longterm = "shortterm"

            tosave[qid] = {
                    "answer": answer,
                    "gt": gt,
                    "qid_objs": q2_obj2id,
                    "img_objs": q_obj2id_save,
                    "fl": fl,
                    "l": layout,
                    "args": obj_args,
                    "arg_ids": id_args,
                    "longterm": longterm,
                    "detailed": detailed,
                    "n_rels": n_rels,
                    "overlap": n_overlap,
                    "rels": filtered_rels,
                    "intersection": bb_overlaps
                    }

    ### end of specific if statement ###
    if split == "val":
        makeJson(tosave, "custom_json/val_balanced_annotations")

makeJson(obj2child, "custom_json/obj2child")
