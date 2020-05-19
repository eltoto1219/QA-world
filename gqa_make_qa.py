from utils import makeJson, makeTxt, loadTxt, loadJson, loadGQAData, FindInParens, getTrueNouns, word2alternatives, getCompoundInList, INVALID_NOUNS,CONVERT, substring, getLCS
from collections import defaultdict, Counter
from tqdm import tqdm
import re
from shapely.geometry import Polygon
import statistics
from nltk import word_tokenize, pos_tag, download
from nltk.corpus import wordnet as wn
import inflect
from copy import deepcopy

PRINT = False
ISGQA = True
relations = loadTxt("phase_1/relations")
attributes = loadTxt("phase_1/attributes")
objects = loadTxt("phase_1/objects")
id2obj = loadJson("phase_1/gqa_id2obj")
if ISGQA:
    graphs = loadJson("phase_1/gqa_img2info")
    splits = ["val", "train"]
    subsets = ["balanced", "all"]
    train_annos = []
    val_annos = []
    all_answers = set()

qid2ambgOverlap =  defaultdict(list)
qid2normOverlap = defaultdict(list)
ambigous_objects =  defaultdict(list)
find = FindInParens(id2obj, ("OBJECTS", objects))
uniq_rels = deepcopy(relations)
uniq_attrs =  deepcopy(attributes)
uniq_objs = deepcopy(objects)

n_with_ambigous = 0
n_with_overlap = 0
n_with_none = 0
total_data = 0

print("start")
for split in splits:
    for subset in subsets:
        if ISGQA:
            questions = loadGQAData(subset, "data_gqa", split)
            total_data += len(questions)
        name = split + "_" + subset + "_items"
        data = []
        for qid, v in tqdm(questions.items()):

            question = v['question'].lower()
            answer = v['answer'].lower()
            all_answers.add(answer)
            if answer in uniq_objs:
                uniq_objs.remove(answer)
            if answer in uniq_attrs:
                uniq_attrs.remove(answer)
            if answer in uniq_rels:
                uniq_rels.remove(answer)
            #try:
            if ISGQA:
                image = v["imageId"]
                img_objs = set([o["name"] for o in graphs[image]])
            #except:
            #    print("FAIL")
            #    continue

            #get overlapping objects from question annotaitons
            if ISGQA:
                fa = v['fullAnswer'].replace(",", "").lower()
                args = [l["argument"] for l in questions[qid]["semantic"]]
                qid_objects = {id2obj[oid].lower(): oid\
                        for oid in v["annotations"]["question"].values()}
                overlap_objs = list(
                        filter(
                            lambda x: x in img_objs and x in question,
                            qid_objects))
                qid2normOverlap[qid] += overlap_objs
            else:
                overlap_objs = []

            #check objects in answer and layout
            matches = set()
            if answer not in "yes no" and\
                    answer not in relations and\
                    answer not in attributes and answer not in INVALID_NOUNS:
                matches.add(answer)
            for a in args:
                obj, a_type = find(a)
                if a_type == "INVALID":
                    for oa in word2alternatives(obj):
                        for oa in img_objs:
                            if oa in a:
                                obj = oa
                                break
                if obj and obj in img_objs:
                    add = True
                    for oa in word2alternatives(obj):
                        if oa in matches:
                            add = False
                    if add:
                        matches.add(obj)


            #get full answer matches
            answer_matches = set()
            for o in img_objs:
                if o in fa:
                    answer_matches.add(o)
                try:
                    if CONVERT[o] in fa:
                        answer_matches.add(o)
                except Exception:
                    pass

            #get all nouns not in  overalpping objects
            nouns = getTrueNouns(question, overlap_objs,
                relations, attributes, objects, qid_objects)
            nouns = [n for n in nouns if n not in overlap_objs and n not in INVALID_NOUNS]

            #if "false" object exists in full answer, remove it
            n_set = set(nouns)
            if "no" in answer or "no" in fa:
                for o in answer_matches:
                    o_set = set(word2alternatives(o))
                    if not o_set.intersection(img_objs)\
                            and o_set.intersection(n_set):
                        try:
                            nouns.remove(o)
                        except Exception:
                            pass

            #again also check LCS of answer and question and compare to answer
            #if not equal number of object, then delete obj from nouns
            dif_obj = False
            c, seq, length, (row, col) = substring(question, fa)
            substr = getLCS(question, c, length, row, col)
            if substr and "no" in answer:
                rest_answer = fa[fa.find(substr):].replace(substr, "")
                rest_of_q = question[question.find(substr):].replace(substr, "")
                no_a = 0
                no_q = 0
                for n in nouns:
                    if n in rest_answer or\
                            (n in CONVERT and CONVERT[n] in rest_answer):
                        no_a += 1
                    if n in rest_of_q or\
                            (n in CONVERT and CONVERT[n] in rest_answer):
                        no_q += 1
                for o in img_objs:
                    if o in rest_answer or\
                            (n in CONVERT and CONVERT[n] in rest_answer):
                        no_a += 1
                    if o in rest_of_q or\
                            (n in CONVERT and CONVERT[n] in rest_answer):
                        no_q += 1

                for n in nouns:
                    if n in rest_of_q and n not in rest_answer and no_a == no_q:
                        nouns.remove(n)
                        if PRINT:
                            print("REMOVING", n)

            # FINALLY MATCHING STAGE store ambigous connections in here
            removed = []
            ambigous = []

            #if object is dfferent sense in question, add here
            for n in nouns:
                for o in img_objs:
                    n_set = set(word2alternatives(n))
                    if o in n_set and\
                        not n_set.intersection(INVALID_NOUNS) and\
                        not n_set.intersection(set(overlap_objs)):
                        if n in nouns:
                            ambigous.append(o)
                            nouns.remove(n)
                            removed.append(n)
                            if o not in ambigous_objects[n]:
                                ambigous_objects[n].append(o)

            # add ambigous cconnections from full answer/just answer
            #this is when obj in question relates to something in answer
            for n in nouns:
                if answer_matches:
                    n_set = set(word2alternatives(n))
                    if not n_set.intersection(answer_matches):
                        ambigous += list(answer_matches)
                        for a in answer_matches:
                            if a not in ambigous_objects[n]:
                                ambigous_objects[n].append(a)
                        nouns.remove(n)
                        removed.append(n)

            #if no objects in full answer (ei reason there are still nouns left)
            #map every object to ambiogus word
            for n in nouns:
                if n not in ambigous and\
                ("no" not in fa  or "no" in answer) and\
                n in fa:
                    for o in img_objs:
                        if o not in ambigous_objects[n]:
                            ambigous_objects[n].append(o)

            #clean just incase now:
            for cand in nouns + removed:
                if not ambigous_objects[cand]:
                    ambigous_objects.pop(cand)
                    if cand in ambigous_objects:
                        ambigous_objects.removed(cand)

            if ambigous:
                qid2ambgOverlap[qid] += ambigous

            all_overlaps = list(overlap_objs) + ambigous

            if overlap_objs:
                n_with_overlap += 1
            if not overlap_objs and ambigous:
                n_with_ambigous += 1
            if not overlap_objs and not ambigous:
                n_with_none += 1

            if PRINT:
                print("AMBG OVERLAP", {n : [ i for i in ambigous_objects[n]\
                        if i in ambigous] for n in nouns + removed if ambigous})
                print("NORM OVERLAP", overlap_objs)
                print("QUESITONS", question)
                print("ANSWER", answer, fa)
                input()

            target = {
                "question_id": qid,
                "image_id": image,
                "question": question,
                "label": answer,
                "overlaps": all_overlaps
                }

            data.append(target)

    if split == "train" and substet == "balanced":
        makeJson(ambigous_objects, "phase_1/gqa_abg_relations")
        makeJson(qid2normOverlap, "phase_1/gqa_qid2normOverlap")
        makeJson(qid2ambgOverlap, "phase_1/gqa_qid2ambgOverlap")
        makeTxt(all_answers, "phase_1/gqa_answers")
        makeTxt(uniq_attrs, "phase_1/uniq_attrs")
        makeTxt(uniq_objs, "phase_1/uniq_objs")
        makeTxt(uniq_rels, "phase_1/uniq_rels")
    makeJson(data, "phase_1/" + name)

print("% normal overlap", (n_with_overlap/total_data)*100)
print("% ambigous overlap", (n_with_ambigous/total_data)*100)
print("% no overlap", (n_with_none/total_data)*100)
