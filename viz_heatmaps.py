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


#load data
plt.rcParams["figure.figsize"] = (20,20)
qid2pred = json.load(open("testdev_predict_aug.json", "r"))
total = len(qid2pred)
valid = json.load(open("/ssd-playpen/avmendoz/lxmert/data/gqa/testdev.json", "r"))
#scenes = json.load(open("/ssd-playpen/avmendoz/lxmert/testdev_sceneGraphs.json", "r"))
questions = json.load(open("/ssd-playpen/avmendoz/lxmert/questions/testdev_balanced_questions.json", "r"))
print("Loaded data")

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

def layout_cooccur(coords, *args):
    seen_inds = set()
    names2ind =  {}
    for d in args:
        #print(d)
        for k, v in d.items():
            if v in seen_inds:
                raise Exception("counted same ind more than once")
            else:
                seen_inds.add(v)
            names2ind[k] = v
    #raise Exception
    size = len(names2ind)
    matrix = np.zeros((size, size))
    assert matrix.shape == tuple([size,size]), "matrix not {} should be {}".format(matrix.shape, tuple([size,size]))
    for c in coords:
        matrix[c[0], c[1]] +=1
    for i in range(0, len(matrix)):
        matrix[:, i] -= matrix[:, i].mean()
        divisor = matrix[:, i].max()
        if divisor:
            matrix[:, i] /= divisor
    return matrix, names2ind

def layout_heatmap(matrix, ind2names, imgname):
    print("plotting: {}".format(imgname))
    plt.imshow(matrix, interpolation='nearest', cmap='Reds')
    plt.yticks(range(len(ind2names)), list(ind2names.keys()), fontsize=14)
    plt.xticks(range(len(ind2names)), list(ind2names.keys()), fontsize=14, rotation=90)
    plt.savefig("figs/{}.png".format(imgname, dpi = 100))

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

def coocmatrix(features, names, class2position, ll):
    cc_len = len(names)
    matrix =  [[0] * cc_len] * cc_len
    matrix = np.array(matrix)
    counts = deepcopy(matrix)
    assert ll > 0, "wtf"

    # make matrix
    for vec in features:
        l = len(vec)
        for row_cls, row_offset in enumerate(vec):
            row = class2position[row_cls] + row_offset
            for col_cls, col_offset in enumerate(vec):
                col = class2position[col_cls] + col_offset
                matrix[row, col] += 1
    cls_change = {v:k for k,v in class2position.items()}
    cls = 0
    #norm-by-class
    print("\t norm")
    for i, row in enumerate(matrix):
        row_vec = []
        sub_vec = []
        cls = 0
        for j, col in enumerate(row):
            if cls + 1 == 6:
                end = len(matrix)
            else:
                end = class2position[cls + 1]
            sub_vec.append(int(matrix[i,j]))
            if j + 1 == end:
                start = class2position[cls]
                dif = end - start
                val = np.linalg.norm(np.array(sub_vec))
                if val != 0:
                    row_vec += [a/val for a in sub_vec]
                else:
                    row_vec = deepcopy(sub_vec)
                sub_vec = []
                cls +=1
        try:
            counts[i,:] = np.array(row_vec)
        except:
            print(i, j, np.array(row_vec).shape)
    return counts

def makeHeatmaps(matrix, class2name, ID):
    ynames = []
    xnames = []
    vals = {v:k for k,v in class2position.items()}
    vals2 = list(vals.keys())
    vals2.append(len(names))
    i = 0
    for end_g in vals2[1:]:
        start_g = class2position[i]
        if i == 5:
            end_g = len(names)
        else:
            end_g = class2position[i+1]
        i += 1
        j = 0
        for inner_e in vals2[1:]:
            print("\tmaking....")
            inner_s = class2position[j]
            if j == 5:
                inner_e = len(names)
            else:
                inner_e = class2position[j+1]
            j +=1
            ynames = [names[ind] for ind in range(start_g, end_g)]
            xnames = [names[ind] for ind in range(inner_s, inner_e)]
            #toplt = np.concatenate((matrix[start_g:end_g, :start_g],
            #matrix[start_g:end_g, end_g:]), axis = 1)
            plt.imshow(matrix[start_g:end_g, inner_s:inner_e],
                    interpolation='nearest', cmap='Reds')
            plt.yticks(range(len(ynames)), ynames, fontsize=14)
            plt.xticks(range(len(xnames)), xnames, fontsize=14, rotation=90)
            plt.savefig("figs/{}/{}-{}.png".format(
                ID, class2name[vals[start_g]], class2name[vals[inner_s]]), dpi = 150)

def checker(layout):
    b = False
    weird = set(["and", "or", "different", "same", "common"])
    for w in weird:
        if w in layout:
            b = True
    return b

#prelimes
qid2ans = {v["question_id"]: v['label'].keys() for v in valid}
qid2img = {v["question_id"]: v['img_id'] for v in valid}
#setup accs
keys = ["location", "global_group", "objs", "structural", "semantic", "detailed", "ops", "layout"]
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
wrong_layouts = []
right_layouts = []
layout_features = []


# main run of the file
for qid in tqdm(qid2pred):
    feat_i =  []
    layout_vec = []
    feat_vec = []
    qID = qid["questionId"]
    pred = qid["prediction"]
    img_id = qid2img[qID]
    ans = qid2ans[qID]
    if pred in ans :
        correct += 1
        result = 1
    else:
        result = 0
    # start sup accs now
    full_layout = "".join([l["operation"] + "-" for l in questions[qID]["semantic"]])[:-1]
    full_args = "".join([l["argument"] + "-" for l in questions[qID]["semantic"]])[:-1]
    sent = questions[qID]["question"] + ":" + str(pred) + ":" + str(ans) + ":" + full_args+ ":" + full_layout
    #try:
    #    location = scenes[img_id]["location"]
    #except KeyError:
    location = "none"
    ind = update("location", location, totals, corrects, sent)
    feat_vec.append(ind) #1
    group = questions[qID]["groups"]["global"]
    ind = update("global_group", group, totals, corrects, sent)
    feat_vec.append(ind) # 2
    """
    list: obs
    anno_objectids = [o for o in questions[qID]["annotations"]["question"].values()]
    for obj in anno_objectids:
    update("objs", obj, totals, corrects)
    """
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

#ogram_accs = {}
#ogram_total = {}
#tgram_accs = {}
#tgram_total = {}
#thgram_accs = {}
#thgram_total = {}
#print("making csv")
#for f, a in tqdm(zip(layout_features, answers)):
#    if a == 1:
#        for o in f[0]:
#            if o not in ogram_accs:
#                ogram_accs[o] = 0
#            else:
#                ogram_accs[o] +=1
#        for o in f[1]:
#            if o not in tgram_accs:
#                tgram_accs[o] = 0
#            else:
#                tgram_accs[o] +=1
#
#        for o in f[2]:
#            if o not in thgram_accs:
#                thgram_accs[o] = 0
#            else:
#                thgram_accs[o] +=1
#
#    for o in f[0]:
#        if o not in ogram_total:
#            ogram_total[o] = 0
#        else:
#            ogram_total[o] +=1
#    for o in f[1]:
#        if o not in tgram_total:
#            tgram_total[o] = 0
#        else:
#            tgram_total[o] +=1
#    for o in f[2]:
#        if o not in thgram_total:
#            thgram_total[o] = 0
#        else:
#            thgram_total[o] +=1
#
#for k, v in ogram_total.items():
#    acc = round(ogram_accs[k]/v * 100, 4)
#    freq = round(v/total * 100, 4)
#    CSV2.append(["1-gram", k, acc, freq])
#for k, v in tgram_total.items():
#    acc = round(tgram_accs[k]/v * 100, 4)
#    freq = round(v/total * 100, 4)
#    CSV2.append(["2-gram", k, acc, freq])
#
#print("three gram", thgram_total)
#for k, v in thgram_total.items():
#    acc = round(thgram_accs[k]/v * 100, 4)
#    freq = round(v/total * 100, 4)
#    CSV2.append(["3-gram", k, acc, freq])
#
#with open('layouts.csv', 'w', newline='') as f:
#        writer = csv.writer(f)
#        writer.writerows(CSV2)
#with open('analysis.csv', 'w', newline='') as f:
#        writer = csv.writer(f)
#        writer.writerows(CSV)
#
# make heatmap
#answers = np.array(answers)
#features = np.array(features)
#wrong = np.array(wrong)
#right = np.array(right)
#ind2class = update.ind2class
#class2ind = update.class2ind
#ops = class2ind.pop("ops")
#objs = class2ind.pop("objs")
### make matrices
#print("making layout coocs")
#rl_matrix, names = layout_cooccur(right_layouts, layouts)
#wl_matrix, names = layout_cooccur(wrong_layouts, layouts)
#dif = abs(rl_matrix - wl_matrix)
#layout_heatmap(rl_matrix, names, "right")
#layout_heatmap(wl_matrix, names, "wrong")
#layout_heatmap(dif, names, "dif")
#print("weve reached the end")

#class2position, class2names, names = reset(ind2class, class2ind)
#print("wrong matrix")
#wrong_matrix = coocmatrix(wrong, names, class2position)
#print("right matrix")
#right_matrix = coocmatrix(right, names, class2position)
#print("dif matrix")
#dif_matrix = wrong_matrix - right_matrix
#dif_matrix = np.where(dif_matrix<0, 0, dif_matrix)
#plt.imshow(dif_matrix,
#                interpolation='nearest', cmap='Reds')
#plt.savefig("figs/{}_heatmap.png".format("dif"), dpi = 300)
#print("wrong")
#makeHeatmaps(wrong_matrix, class2names, "wrong")
#print("right")
#makeHeatmaps(right_matrix, class2names, "right")
#print("dif")
#makeHeatmaps(dif_matrix, class2names, "dif")
#with open('boo.csv', 'w', newline='') as f:
#        writer = csv.writer(f)
#        writer.writerows(matrix)

#print("start modeling")
#for i in range(6):
#    model = CategoricalNB()
#    f_i = features[:, i]
#    f_i = f_i[:,np.newaxis]
#    model.fit(f_i, answers)
#    print(model.score(f_i, answers))
#model = CategoricalNB()
#model.fit(features[:, -2:], answers)
#print(model.score(features[:, -2:], answers))
