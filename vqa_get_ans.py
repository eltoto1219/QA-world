from utils import loadJson, loadTxt, makeTxt, makeJson, getTrueNouns, word2alternatives, union, intersect, IGNORE
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
from copy import deepcopy
import math
import re
from collections import defaultdict
from num2words import num2words

MIN = 6


train_data = []
val_data = []

#data to make
ans_count = Counter()
ans2num = {}
num_ans = 0
total_trash = 0
total_data = 0
qid2num = {}
qid2keep = {}

#data to write
vqa_number_subset_qids = []
vqa_train_qid2ans = []
vqa_val_qid2ans = []

vqa_trashed_qids = set()
vqa_gqa_ans_overlap = set()
uniq_ans = set()

#data to load
uniq_attributes = loadTxt("phase_1/uniq_attrs")
uniq_relations = loadTxt("phase_1/uniq_rels")
uniq_objects = loadTxt("phase_1/uniq_objs")
uniq = union(uniq_attributes, uniq_relations, uniq_objects)
gqa_answers = loadTxt("phase_1/gqa_answers")

#funcs

def keepCompound(word):
    aug = set(word.split(" "))
    if len(aug) > 1:
        if aug.union({"and", "or", "no", "yes", "-", "'", "/"}):
            return False
        else:
            return  word
    else:
        return word

def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1

def isOrdinal(word):
    og = word
    if len(word.split(" ")) > 1:
        return word
    else:
        word = re.findall('\d+', word)
        if not word:
            return og
        if len(word) > 1:
            return False
        elif og[-2:] in ["st" "nd", "rd", "th"]:
            word = word[0]
            return num2words(int(word), ordinal =True)
        else:
            return False

#data to make
for z in ["val"]:
    data = loadJson("data_vqa/v2_mscoco_{}2014_annotations".format(z))["annotations"]
    total_data += len(data)
    if z == "train":
        train_data = data
    else:
        val_data = data
    for a in tqdm(data):
        label2score = {}
        qid = a["question_id"]
        img_id = a["image_id"]
        a_type = a["answer_type"]
        gt = a["multiple_choice_answer"].lower()
        ans_count[gt] += 1
        if gt not in ans2num:
            ans2num[gt] = num_ans
            num_ans += 1
        answers = Counter()
        for n in a["answers"]:
            an =  n["answer"].lower()
            answers[an] += 1
        for a in answers:
            ans_count[a] += 1
        cands = list(filter(lambda x: x in union(gqa_answers, uniq), word2alternatives(gt)))
        #if not in gqa and question is a number question
        if a_type == "number" and not cands:
            answers = {k:v for k,v in answers.items() if k.isdigit() and k != gt}
            if gt.isdigit():
                label2score[gt] = 1
            for a in answers:
                if a not in ans2num:
                    ans2num[a] = num_ans
                    num_ans += 1
                label2score[a] = get_score(answers[a])
            if not label2score:
                vqa_trashed_qids.add(qid)
                total_trash += 1
            else:
                qid2num[qid] = label2score
        else:
            #gt check
            for c in cands:
                if c in gqa_answers:
                    vqa_gqa_ans_overlap.add(gt)
                    label2score[gt] = 1
                    break
                elif isOrdinal(c) and  keepCompound(c) and c not in IGNORE:
                    uniq_ans.add(isOrdinal(gt))
                    label2score[gt] = 1
                    break
            # other answers check
            accept = list(filter(lambda x: x[0] not in label2score, answers.items()))
            for (a,f) in accept:
                if a in gqa_answers:
                    vqa_gqa_ans_overlap.add(a)
                    label2score[a] = get_score(f)
                else:
                    for w in word2alternatives(a):
                        if isOrdinal(w) and keepCompound(w) and w not in IGNORE:
                            uniq_ans.add(isOrdinal(a))
                            if a not in ans2num:
                                ans2num[a] = num_ans
                                num_ans +=1
                            label2score[a] = get_score(f)
                            break
            if not label2score:
                vqa_trashed_qids.add(qid)
            else:
                qid2keep[qid] = label2score

for split, st, in zip([val_data, train_data], ["val", "train"]):
    for d in split:
        qid = d["question_id"]
        img_id = d["image_id"]
        if qid in vqa_trashed_qids:
            pass
        elif qid in qid2keep:
            target = {
                'question_id': qid,
                'image_id': img_id,
                'answer': {isOrdinal(a): v for a, v in qid2keep[qid].items() if ans_count[a] >= MIN},
                }
            if st == "val":
                vqa_val_qid2ans.append(target)
            else:
                vqa_train_qid2ans.append(target)
        elif qid in qid2num:
            target = {
                'question_id': qid,
                'image_id': img_id,
                'answer': {isOrdinal(a): v for a, v  in qid2num[qid].items() if ans_count[a] >= MIN},
                }
            vqa_number_subset_qids.append(target)

datas = [
        ans2num,
        vqa_number_subset_qids,
        vqa_trashed_qids,
        vqa_train_qid2ans,
        vqa_val_qid2ans,
        uniq_ans
        ]
files = [
        ("vqa_ans2id", "json"),
        ("vqa_number_subset_qids", "json"),
        ("vqa_trashed_qids", "txt"),
        ("vqa_train_qid2ans", "json"),
        ("vqa_val_qid2ans", "json"),
        ("vqa_uniq_ans", "txt")
        ]

for d, (f, t) in zip(datas, files):
    if t == "json":
        makeJson(d, "phase_2/" + f)
    elif t == "txt":
        makeTxt(d, "phase_2/" + f)
    else:
        raise Exception

per = (total_trash/total_data)*100
print("new answers", len(uniq_ans))
print("overlap", len(vqa_uniq_ans))
print("p-removed", per)
