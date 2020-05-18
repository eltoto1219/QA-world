import numpy as np
import json
import csv
import sklearn as sk
from sklearn.naive_bayes import CategoricalNB
from sklearn.cluster import KMeans
from copy import deepcopy
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import functions

def getInParens(inp):
    return int(re.search('\(([^)]+)', inp).group(1))

#Classes + Functions
class Update:
    def __init__(self):
        self.class_ind = {k: {} for k in keys}
        self.class2ind = {k: {} for k in keys}
        self.ind2class = {k: {} for k in keys}
        self.sents = {k: {} for k in keys}

    def __call__(self, key, name, totals, corrects, sent):
        if name not in corrects[key]:
            corrects[key][name] = result
            totals[key][name] = 1
            self.sents[key][name] = sent
        else:
            corrects[key][name] += result
            totals[key][name] += 1

        if name not in self.class2ind[key]:
            ind = len(self.class_ind[key])
            self.class2ind[key][name] = ind
            self.ind2class[key][ind] = name
            self.class_ind[key][name] = ind
        return self.class_ind[key][name]

def reset(ind2class, class2ind):
    class2position =  {}
    position = 0
    names = []
    class2name = {}
    for cls_i, c in enumerate(class2ind):
        class2position[cls_i] = position
        class2name[cls_i] = c
        offset = 0
        for sc in class2ind[c].keys():
            names.append(str(c) + "-" + str(sc))
            position += 1
    return class2position,  class2name, names


# main run of the file
def checker(layout):
    b = False
    weird = set(["and", "or", "different", "same", "common"])
    for w in weird:
        if w in layout:
            b = True
    return b

def getAllLayoutGrams():
    structural = questions[qID]["types"]["structural"]
    ind = update("structural", structural.split(" ")[0], totals, corrects, sent)
    feat_vec.append(ind) #3
    semantic = questions[qID]["types"]["semantic"]
    ind = update("semantic", semantic.split(" ")[0], totals, corrects, sent)
    feat_vec.append(ind) #4
    detailed = questions[qID]["types"]["detailed"]
    ind = update("detailed", detailed.split(" ")[0], totals, corrects, sent)
    feat_vec.append(ind) #5
    #list: ops
    individual_ops = set([o["operation"].split(" ")[0] for o in questions[qID]["semantic"]])
    for op in individual_ops:
        update("ops", op, totals, corrects, sent)
    layout = [o["operation"].split(" ")[0] for o in questions[qID]["semantic"]]
    layout = "".join(l + "-" for l in layout)[:-1]
    ind = update("layout", layout, totals, corrects, sent)
    feat_vec.append(ind) #6
    features.append(feat_vec)

    #layout parser
    full_layout = layout
    weird = set(["and", "or", "different", "same", "common"])
    split_full = full_layout.split("-")
    bi_gram_parse = []
    tri_gram_parse = []
    weird_check = checker(full_layout)
    #monogram
    mono_gram_parse = list(set(split_full))
    #bigram
    for i in range(0, len(split_full)-1):
        if split_full[i+1] == "select" or split_full[i+1] in weird:
            pass
        else:
            bi = split_full[i] + "-" + split_full[i+1]
            for w in weird:
                assert w not in bi
            bi_gram_parse.append(bi)
    bi_gram_parse = list(set(bi_gram_parse))
    #trygram
    if len(split_full) > 2:
        if "select-select" in full_layout:
            tri_gram_parse.append("select-select-" + split_full[-1])
        else:
            if "exist" in full_layout and weird_check:
                tri_gram_parse.append("exist-exist-and")
            elif "verify" in full_layout and weird_check:
                tri_gram_parse.append("verify-veriy-and")
        for i in range(0, len(split_full)-2):
            if split_full[i+1] == "select" or split_full[i+2] == "select" or split_full[i+2] in weird:
                pass
            else:
                tri = split_full[i] + "-" + split_full[i+1] + "-" + split_full[i+2]
                for w in weird:
                    assert w not in tri
                tri_gram_parse.append(tri)
    for m in mono_gram_parse:
        if m not in monogram_layouts:
            monogram_layouts[m] = monogram
            monogram += 1
            layouts[m] = omni
            omni +=1
    for b in bi_gram_parse:
        if b not in bigram_layouts:
            bigram_layouts[b] = bigram
            layouts[b] = omni
            bigram += 1
            omni +=1
    for t in tri_gram_parse:
        if t not in trigram_layouts:
            trigram_layouts[t] = trigram
            layouts[t] = omni
            trigram += 1
            omni +=1
    for m in mono_gram_parse:
        mi = layouts[m]
        for b in bi_gram_parse:
            bi = layouts[b]
            layout_vec += [tuple([mi, bi]), tuple([bi, mi])]
            for t in tri_gram_parse:
                tri = layouts[t]
                layout_vec += [tuple([mi, tri]), tuple([tri, mi])]
                layout_vec += [tuple([bi, tri]), tuple([tri, bi])]
    if result == 0:
        wrong.append(feat_vec)
        wrong_layouts +=  layout_vec
    else:
        right.append(feat_vec)
        right_layouts += layout_vec
    feat_i = [mono_gram_parse, bi_gram_parse, tri_gram_parse]
    layout_features.append(feat_i)
    answers.append(result)


def makeCSV(totals, corrects):
    CSV = [["class", "subclass", "acc", "freq", "sent", "pred", "answer", "args", "full-ops"]]
    CSV2 = [["n-gram", "acc", "acc", "freq"]]
    accs = {k: {} for k in keys}
    for (tk, tv), (ck, cv) in zip(totals.items(), corrects.items()):
        temp = []
        for (tk2, tv2), (ck, cv2) in  zip(tv.items(), cv.items()):
            sent = update.sents[tk][tk2]
            acc = round(cv2/tv2 * 100, 4)
            freq = round(tv2/total * 100, 4)
            to_append = [tk, tk2, acc, freq, sent]
            to_append += sent.split(":")
            temp.append(to_append)
            accs[tk][tk2] = {"acc": acc, "freq": freq}
        temp.sort(key=lambda x:x[2])
        CSV.extend(temp)

    print("making csv")
    with open('layouts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(CSV2)

if __name__ == "__main__":
# ACTUAL FILE BELOW #

    BASE = "/ssd-playpen/avmendoz/gqa_analysis"

    #load data
    questions = json.load(os.path.join(BASE, "data","testdev_balanced_questions.json"))
    og = json.load(open(os.path.join(BASE, "data", "submit_predict_test.json", "r")))
    aug = json.load(open(os.path.join(BASE, "questions", "testdev_predict_aug.json", "r")))
    assert len(questions) == len(og) == len(aug)
    print("Loaded data")

    #setup the data
    qid2info = {
            k: {
                "answer": v['answer'],
                "question": v['question'],
                "fullAnswer": v['fullAnswer'],
                "group": v['group'],
                "structural": v["types"]["structural"],
                "semantic": v["types"]["semantic"],
                "detailed": v["types"]["detailed"],
                "fullLayout": "".join([l["operation"] + "-"\
                        for l in questions[qID]["semantic"]])[:-1],
                "args": "".join([getInParens(l["argument"]) + "-"\
                        for l in questions[qID]["semantic"]])[:-1],
                "layout": "".join([l["operation"].split(" ")[0] + "-"\
                        for l in questions[qID]["semantic"]])[:-1],
                }
            for k, v in questions.items()}


    corrects = {k: {} for k in keys}
    totals = {k: {} for k in keys}
    #setup dict
    correct = 0
    #feature order
    features = []
    right = []
    wrong = []
    answers = []
    update = Update()
    print("----")
    ll = len(qid2pred)
    #create dicts for layouts
    layouts = {}
    monogram_layouts = {}
    monogram = 0
    bigram_layouts = {}
    bigram = 0
    trigram_layouts = {}
    trigram = 0
    omni = 0

    for o, a in in tqdm(zip(og aug)):
        assert o["questionId"] == a["questionId"]
        qid = o["questionId"]
        o_pred = o["prediction"]
        a_pred = a["prediction"]
        img_id = qid2img[qID]
        if pred in ans :
            correct += 1
            result = 1
        else:
            result = 0
