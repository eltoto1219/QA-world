import torch
import numpy as np
import json
import csv
from sklearn.naive_bayes import CategoricalNB
from sklearn.cluster import KMeans
from copy import deepcopy
import scipy.sparse as sp
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag, download
import re
import sys
from functools import lru_cache
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
#import pattern3.en as pattern3
import inflect
from num2words import num2words

#make inflection engine
PRINT = False
ENGINE = inflect.engine()

#conditions for data
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
    for name, color in colors.items())
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
        '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!']

#combinatin of pos in wordnet and ones by me
CONVERT = {"male": "man", "female": "woman", "woman": "female", "man": "male", "women": "female", "girl": "female", "boy": "male", "men": "male", "she": "woman", "he": "man", "her": "woman", "his": "man"}
INVALID_NOUNS = {"of", "in", "on", "side", "than", "left", "right", "look", "by", "as", "image", "picture", "color", "scene", "gender"}
POS = ["a", "s", "r", "n", "v", "d", "t", "p"]
FORBIDDEN = ["left", "right"]
ADJ, ADJ_SAT, ADV, NOUN, VERB, DETERMINER, ARTICLE, POSITIONAL = POS
POSMAP = {
        "CC": False,
        "CD": False,
        "DT": ARTICLE,
        "EX": DETERMINER,
        "FW": False,
        "IN": False,
        "JJ": ADJ,
        "JJR": ADJ,
        "JJS": ADJ,
        "LS": False,
        "MD": DETERMINER,
        "NN": NOUN,
        "NNS": NOUN,
        "NNP": NOUN,
        "NNPS": NOUN,
        "PDT": DETERMINER, #an object, or subject but group of words
        "POS": NOUN,
        "PRP": False,
        "PRP": False,
        "RB": ADV,
        "RBR": ADV,
        "RBS": ADV,
        "RP": False,
        "TO": False,
        "UH": False,
        "VB": VERB,
        "VBD": VERB,
        "VBG": VERB,
        "VBN": VERB,
        "VBP": VERB,
        "VBZ": VERB,
        "WDT": DETERMINER, #wh-determiner which
        "WP": DETERMINER, #wh-pronoun who, what
        "WP$": DETERMINER, #possessive wh-pronoun
        "WRB": DETERMINER, #wh-adverb where when
        ".": False,
        ":": False,
        ",": False,
        "PRP$": False,
        "`": False,
        "``":False
        }
IGNORE = [
'january',
'february',
'march',
'april',
'may',
'june',
'july',
'august',
'september',
'october',
'november',
'december'
"ge money",
"cnn",
"m&m's",
"nsp",
"ty",
"ihop",
"lg",
"vw",
"at&t",
"ibm",
"gmc",
"the",
"bnsf",
"htc",
"lsu",
"csx",
"django",
"sasa",
"easo",
"klm",
"atm",
"kfc",
"nowhere",
"none",
"am",
"noon",
"yes",
"no",
"ny",
"pm",
"hp",
"soon",
"dc",
"dachshund",
"thomas",
"us",
"replace",
"pc",
"db",
"alto",
"abc",
"boa",
]

# NLTK functions

def existPersonalPronoun(d_pos, sent):
    d_pos = d_pos[:-1] if d_pos[-1] == "S" else d_pos
    unique_d_pos = set(list(tagged["dpos"].values()))
    if "NNP" == d_pos and "NNP" in unique_d_pos\
            or "NNPS" in unique_d_pos:
        return  True
    elif "NNP" != d_pos and "NNP" in unique_d_pos\
            or "NNPS" in unique_d_pos:
        return False
    elif "NNP" != d_pos and "NNP" not in unique_d_pos\
            or "NNPS" not in unique_d_pos:
        return True
    elif "NNP" == d_pos and "NNP" not in unique_d_pos\
            or "NNPS" not in unique_d_pos:
        return False
    else:
        raise Exception


# WORDNET FUNCTIONS

def synsetWordEquality(word_q, word_k):
    if word_k in word_q or word_q in word_k or wn.morphy(word_k) == wn.morphy(word_q):
        return True
    else:
        False

def cleanSynsetDefinition(def_sent):
    x = def_sent.replace("(", ". ")
    x = x.replace(")", "")
    x = x.replace("-", " ")
    return x

def synset2String(word, notFound = None):
    if type(word) == str:
        return word
    else:
        if notFound is not None:
            notFound.add(word)
        return str(word)[8:].split(".")[0]

def typeChecker(word, r_synset=True):
    if type(word) is str and not r_synset:
        return word
    elif type(word) is str and r_synset:
        #policy: return first avail synset
        return toSynset(word)
    elif type(word) is not str and r_synset:
        #policy: return first avail synset
        return word
    else:
        return toStr(word)

def getSynsets(word, pos = None):
    word = typeChecker(word, r_synset=False)
    if pos is None:
        return wn.synsets(word)
    elif type(pos) is list:
        s = []
        for p in pos:
            s+= wn.synsets(word, pos=p)
        return s
    elif type(pos) is str:
        return wn.synsets(word, pos=pos)

def matchSynsetDefintionPOS(word, d_pos, synset, word_in_def):
    match_def = False
    def_sent = cleanSynsetDefinition(synset.definition())
    d_pos_list = []
    tagged, tokenized = sentWordType(def_sent)
    if dposMatchPersonalNoun(d_pos, tagged) and len(tagged["lookup"]) > 0:
        if word in def_sent:
            for def_sent_i, d_pos_def in tagged["dpos"].items():
                def_word = tokenized[def_sent_i]
                if synsetWordEquality(word, def_word):
                    if equalityDpos(d_pos_def, d_pos):
                        ind_def = True
                        d_pos_list.append(d_pos_def)
            return match_def, d_pos_list, def_sent
        else:
            return True, d_pos_list, def_sent
    return False, d_pos_list, def_sent

def makeSynonymData(data: dict, objects: set, pos=None):
    for o in objects:
        if len(o.split(" ")) > 1:
            continue
        syns = getAllSynonyms(o, pos = pos)
        updateDictList(syns, o, data)

class ObjectMap:
    def __init__(self, load_json="obj_map"):
        if load_json == "obj_map":
            load_json = os.path.join(os.getcwd(), load_json + ".json")
            print(load_json, os.path.isfile(load_json))
        self.path = load_json
        self.min_similarity = 0.1
        self.obj_delim = "-"

        if load_json is None or not os.path.isfile(load_json):
            self.parent_dict = {}
            self.not_found = {}
            self.single_parents = {}
            self.children = {}
            data = self.__load()
            if load_json is None:
                self.path = "obj_map"
            self.__makeMap()
        else:
            data = self.__load()
            print("LOADED: obj map")

    @lru_cache(maxsize=150)
    def getRelative(self, obj, parent = True):
        key = obj
        alt_key = wn.morphy(obj)
        #choose dict
        if parent:
            relatives = self.single_parents
        else:
            relatives = self.children
        #if parent = True -> s = siblings | if parent = False --> s = children
        for r, s in relatives.items():
            #print("MUST BE HERE", r, s)
            if key in r or alt_key in s:
                return r, list(s.keys())
        return False, False

    @lru_cache(maxsize=50)
    def getClosestParent(self, obj1, obj2):
        key = self.__makeKey(obj1, obj2)
        alt_key = self.__makeKey(wn.morphy(obj1), wn.morphy(obj2))
        for p, children in self.parent_dict.items():
            if key in children or alt_key in p:
                return p.replace("'", "")
        return None

    @lru_cache(maxsize=50)
    def getChildrenPairs(obj):
        if obj not in self.parent_dict:
            obj = wn.morphy(obj)
            if obj not in self.parent_dict:
                return []
        else:
            pairs = []
            for pair in self.parent_dict[obj]:
                pair = tuple(self.__parseKey())
                pairs.append(pair)
        return pairs

    def getUnfound(self):
        return self.not_found

    def __makeMap(self):
        for o in tqdm(OBJECTS):
            if len(o.split(" ")) == 1 and o not in self.not_found:
                #make 1x1 dicts
                parents = getParent(o)
                childs = getChild(o)
                for p in parents:
                    p = p.replace("'", "").split("_")[-1]
                    if p not in self.single_parents:
                        self.single_parents[p] = {o: True}
                    else:
                        c = c.replace("'", "")
                        self.single_parents[p][o] = True
                for c in childs:
                    c = c.replace("'", "").split("_")[-1]
                    if c not in self.children:
                        self.children[c] = {o: True}
                    else:
                        self.children[c][o] = True
                #make 2x1 dicts
                for o_other in OBJECTS:
                    if len(o_other.split(" ")) == 1 and o_other != o and o_other not in self.not_found:
                        key = self.__makeKey(o, o_other)
                        o_s = toSynset(o, self.not_found)
                        o_other_s = toSynset(o_other, self.not_found)
                        if o_s and o_other_s:
                            sim = wn.path_similarity(o_s, o_other_s)
                            if sim is None or sim >= self.min_similarity:
                                parent = getLowestCommonParent(o_s, o_other_s)
                                parent = typeChecker(parent, r_synset=False)
                                if parent not in self.parent_dict:
                                    self.parent_dict[parent] = {key: sim}
                                else:
                                    self.parent_dict[parent][key] = sim
        data = {
                "NA": self.not_found,
                "parents": self.parent_dict,
                "all_children": self.children,
                "single_parents": self.single_parents
                }
        self.path = self.path.replace("json", "")
        makeJson(data, self.path)

    def __makeKey(self, obj1, obj2):
        obj1, obj2 = sorted([obj1, obj2])
        return obj1 + self.obj_delim + obj2

    def __parseKey(self, obj):
        return obj.split(self.obj_delim)

    def __load(self):
        self.path = self.path.replace(".json", "")
        data = loadJson(self.path)
        self.parent_dict = data["parents"]
        self.not_found = data["NA"]
        self.children = data["all_children"]
        self.single_parents = data["single_parents"]

# FILE LOADING FUNCTIONS

def loadJson(fname):
    with open("{}.json".format(fname), "r") as f:
        return json.load(f)

def loadTxt(fname, rset = True):
    contents = []
    with open("{}.{}".format(fname, "txt"), "r") as f:
        for row in f:
            row = row.replace("\n", "")
            contents.append(row)
    if rset:
        contents = set(contents)
    return contents

def makeTxt(iterable, fname, wset = False):
    if wset:
        iterable = list(set(iterable))
    with open('{}.txt'.format(fname), 'w', newline='') as f:
        for row in iterable:
            if type(row) is not str:
                row = str(row)
            if "\n" in row:
                f.write(row)
            else:
                f.write(row + "\n")

def makeJson(iterable, fname, vset = False):
    with open('{}.json'.format(fname), 'w', newline='') as f:
        json.dump(iterable, f)

def loadLXMERTData(path = "/ssd-playpen/avmendoz/lxmert/data/gqa"):
    path += 1
    split = json.load(open(path + "valid.json", "r"))
    split += json.load(open(path + "train.json", "r"))
    split += json.load(open(path + "testdev.json", "r"))
    return split

def loadGQAData(split_subset, base_path, *args):
    base_path += "/"
    data = {}
    for arg in tqdm(args):
        if arg == "testdev":
            t = "balanced"
        else:
            t = split_subset
        try:
            splt = json.load(open( base_path + "{}_{}_questions.json".format(arg,t), "r"))
            data = {**data, **splt}
        except FileNotFoundError:
            print("WARNING: split ({}) not found".format(arg))
    assert len(data) != 0, "... no data loaded ..."
    return data

def loadGQAGraphs(base_path, *args):
    base_path += "/"
    data = {}
    for arg in tqdm(args):
        splt = json.load(open(base_path +"{}_sceneGraphs.json".format(arg), "r"))
        data.update(splt)
    assert len(data) != 0, "... no data loaded ..."
    return data

# DATA STRUCTURE FUNCTIONS

def setifyDict(data):
    for d in data:
        data[d] = list(set(data[d]))

def longestInteger(ss):
    try:
        return ss[ss.find("(")+1:ss.find(")")]
    except Exception:
        return ss

def stripEmptyKeys(data):
    for k in data:
        if type(data[k]) == dict:
            try:
                data[k].pop("")
            except:
                pass

def union(*args)  -> set:
    group = []
    for a in args:
        if type(a) == list:
            a = set(a)
        elif type(a) == set:
            pass
        elif type(a) == dict:
            a = set(a.keys())
        else:
            raise TypeError
        group.append(a)
    return set.union(*group)

def intersect(*args)  -> set:
    group = []
    for a in args:
        if type(a) == list:
            a = set(a)
        elif type(a) == set:
            pass
        elif type(a) == dict:
            a = set(a.keys())
        else:
            raise TypeError
        group.append(a)
    return set.intersection(*group)



# GENERAL FUNCTIONS

def compound2ignore(word):
    word = " " + word + " "
    if word in {" and ", " or ", " no ", "yes", "-", "'", "/"}:
        return True
    return False

def fromOrdinal(word):
    og = word
    if not word:
        return False
    elif len(word.split(" ")) > 1:
        return False
    else:
        word = re.findall('\d+', word)
        if not word:
            return False
        if len(word) > 1:
            return False
        elif og[-2:] in ["st" "nd", "rd", "th"]:
            word = word[0]
            return num2words(int(word), ordinal =True)
        else:
            return False




def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
            or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def getContiguous(pairs, sent):
    #chunk to slice (str repr)
    sorted_pairs = sorted([(ind, word) for ind, word in pairs.items()], key=lambda x: x[0])
    chunk2slice = {}
    chunk = ""
    length = len(pairs) - 1
    chunk = ""
    for it, (ind, word) in enumerate(sorted_pairs):
        #make start if empty
        if not chunk:
            start = ind
            prev_ind = ind
        #add word if doesnt break contg
        if abs(ind-prev_ind) <= 1 :
            chunk += word + " "
        #create start if making new chunk
        if abs(ind-prev_ind) > 1 or it == length:
            end = ind
            chunk = chunk[:-1]
            #contg end
            if end == prev_ind + 1:
                slc =slice(start, end+1)
                chunk2slice[chunk] = slc
            #non-contg end
            else:
                #non-contig past
                if prev_ind == start:
                    slc = slice(start,start+1)
                    chunk2slice[chunk] = slc
                    assert chunk == sent[slc][0]
                else:
                    #contig past
                    slc =slice(start, prev_ind+1)
                    chunk2slice[chunk] = slc
            #restart chunk
            start = end
            chunk = "{} ".format(word)
        #make new prev
        prev_ind = ind
    return chunk2slice

# SPECIFIC FUNCTIONS

def posTagSent(sent):
    try:
        tokenized = word_tokenize(sent)
    except LookupError:
        download('punkt')
        tokenized = word_tokenize(sent)
    try:
        tagged = pos_tag(tokenized)
    except LookupError:
        download('averaged_perceptron_tagger')
        tagged = pos_tag(tokenized)
    return tagged

def getNouns(sent):
    return {i:p[0] for i, p in enumerate(posTagSent(sent)) if p[-1] in POSMAP and POSMAP[p[-1]] == NOUN}

def isBinary(ans):
    if "yes" in ans or "no" in ans:
        return True
    return False

def getAttributesAndRelations(scene_i, obj2id, id2obj, attributes, objects, relations, objid2relations, objid2attributes, objname2relations, objname2attributes):
    for obj_id, obj in scene_i["objects"].items():
            obj_name = obj["name"]
            objects.add(obj_name)
            obj2id[obj_name] = obj_id
            id2obj[obj_id] = obj_name
            #make attributes info
            for att in obj["attributes"]:
                attributes.add(att)

                if obj_id not in objid2relations:
                    objid2attributes[obj_id] = set([att])
                else:
                    objid2attributes[obj_id].add(att)

                if obj_name not in objname2attributes:
                    objname2attributes[obj_name] = set([att])
                else:
                    objname2attributes[obj_name].add(att)

            #make realtions info
            for related_info in obj["relations"]:
                relation = related_info["name"]
                obj_related = related_info["object"]
                relations.add(relation)

                if obj_id not in objid2relations:
                    objid2relations[obj_id] = set([relation])
                else:
                    objid2relations[obj_id].add(relation)

                if obj_name not in objname2relations:
                    objname2relations[obj_name] = set([relation])
                else:
                    objname2relations[obj_name].add(relation)

def word2alternatives(word):
    clean = word.replace("'", "").lower()
    candidates = [word]
    if clean not in candidates:
        candidates.append(clean)
    word_m = wn.morphy(word)
    #word_l = pattern3.lemma(word)
    if word_m and word_m not in candidates and word_m:
        candidates.append(word_m)
    #if word_l not in candidates and word_l:
    #    candidates.append(word_l)
    other = []
    for c in candidates:
        try:
            s = ENGINE.singular_noun(c)
        except Exception:
            s = False
        try:
            p = ENGINE.plural(c)
        except Exception:
            p = False
        if s and s not in candidates:
            other.append(s)
        if p and p not in candidates:
            other.append(p)
    return candidates + other


def getTrueNouns(sent, overlap_objs,
        relations, attributes, objects, qid_objects):
    len_sent = len(sent.split(" "))
    nouns = getNouns(sent)
    if PRINT:
        print("PRE NOUNS", nouns)
    compound_inds = []
    cur_r = min(5, len_sent -1)
    #glue together words that exist as compound words
    while(cur_r > 1):
        cw = getCompoundInList(
                sent.replace("?", "").split(" "), sent, objects, radius = cur_r)
        if cw:
            compound_inds += cw
        cur_r -= 1
        if PRINT:
            print("WOOLA obj", cw)
    cur_r = min(5, len_sent -1)
    #glue together words that exist as compound relations
    while(cur_r >= 1):
        cw = getCompoundInList(
                sent.replace("?","").split(" "), sent, relations, radius = cur_r)
        if PRINT:
            print("WOOLA", cw)
        if cw:
            compound_inds += cw
        cur_r -= 1
    if PRINT:
        print("COMPOUND", compound_inds)
    #delete compound objects/relations
    if compound_inds:
        for (cw, c_inds) in compound_inds:
            for ind in c_inds:
                if ind in nouns:
                    nouns.pop(ind)
            #if cw in overlap_objs or cw in relations:
            #    cw_l = len(c_inds)
            #    s = 0
            #    e  = cw_l
            #    found = False
            #    while(e <= len(nouns)):
            #        seq == " ".join(nouns)
            #        if seq == cw:
            #            found = True
            #            for i in range(s, e):
            #                nouns.pop(i)
            #            break
            #        s += 1
            #        e += 1
            #    if not found:
            #        print("THIS IS THE THING NOT FOUND", seq, "REALLY", cw)
    if PRINT:
        print("PRE LEFT", nouns)

    #now check other last conditions
    nouns = deepcopy(list(nouns.values()))
    for i, cand_n in enumerate(nouns):
        if cand_n in relations or\
        cand_n in attributes or\
        cand_n in INVALID_NOUNS:
            nouns.pop(i)
    if PRINT:
        print("MID LEFT", nouns)

    for i, cand_n in enumerate(nouns):
        for cand_alt in word2alternatives(cand_n):
            if cand_alt in overlap_objs or cand_n in overlap_objs:
                if cand_alt in nouns:
                    nouns.remove(cand_alt)
                if cand_n in nouns:
                    nouns.remove(cand_n)
    if PRINT:
        print("POST LEFT", nouns)
    return nouns

def getCompoundInList(list2check, sent2check, set2check, radius = 5):
    if type(list2check)  is dict:
        list2check = list(list2check.key())
    compoundAndInds = []
    s = 0
    e = s + radius
    while(e!=len(list2check)+1):
        sequence = list(range(s, e+1))
        seq_str = " ".join(list2check[s:e])
        if seq_str in set2check and seq_str in sent2check:
            compoundAndInds.append((seq_str, sequence))
        else:
            pass
        s +=1
        e +=1
    return compoundAndInds

def getStuffComplex(tagged, tokenized, rel2ind, unseen_inds, radius = 5, objs = False, obj2action = None, action2prefix = None, action2affix = None, bank=None, obj2article_idx=None, attr=False):
    s = 0
    e = s + radius
    while(e!=len(tokenized)):
        slc = slice(s,e)
        rng = set([i for i in range(s, e)])
        if not rng:
            return
        if rng.issubset(unseen_inds):
            phrase = slc2phrase(tokenized, slc)
            if phrase in bank and not objs and tokenized[slc] and not attr:
                #if tagged["pos"][s] != "a":
                while(True):
                    if phrase in rel2ind:
                        phrase += "$"
                    else:
                        rel2ind[phrase] = slc
                        break
                removeSlc(unseen_inds, slc)
            elif not objs and attr:
                if phrase in bank:
                    rel2ind[phrase] = slc
                elif s in tagged["pos"] and tagged["pos"][s] == "a":
                    rel2ind[phrase] = slc
            elif objs:
                valid = False
                if phrase in WORDNETOBJS or phrase in bank:
                    valid = True
                if len(phrase.split(" ")) == 1 and s in tagged["dpos"] and tagged["dpos"][s] == "NN":
                    valid = True
                if len(phrase.split(" ")) == 1 and s in tagged["dpos"] and tagged["dpos"][s] == "DT" and s-1 in tagged["determiners"]:
                    valid = True
                if phrase in FORBIDOBJS:
                    valid = False
                if valid:
                    fi = slice2firstInd(slc)
                    if tagged["pos"][fi- 1] == "a" and fi -1 in unseen_inds:
                        slc_adj = slice(fi-1,fi)
                        if slc2phrase(tokenized, slc_adj) not in phrase:
                            phrase = slc2phrase(tokenized, slc_adj) + " " + phrase
                        slc = slice(fi-1, e)
                    rel2ind[phrase] = slc
                    removeSlc(unseen_inds, slc)

                    objAction(
                        slice2lastInd(slc)-1, obj2action, action2prefix,
                        action2affix, tagged, tokenized, unseen_inds)

                    if fi - 1 >= 0 and fi -1 in tagged["articles"]:
                        slc2 = slice(fi-1,fi)
                        obj2article_idx[phrase] = slc2
                        tagged["articles"].pop(fi-1)
                        removeSlc(unseen_inds, slc2)
                    if fi -2 >= 0 and tagged["pos"][fi -2] != "v" and tagged["pos"][fi -2] != "n" and tagged["pos"][fi -2] != "a":
                        fi -= 2
                        if fi >= 0 and fi in tagged["articles"]:
                            slc2 = slice(fi,fi+1)
                            obj2article_idx[phrase] = slc2
                            tagged["articles"].pop(fi)
                            removeSlc(unseen_inds, slc2)
        s+=1
        e+=1

# CUSTOM CLASSES

class FindInParens:
    def __init__(self, intIDs, *args):
        """
            args must be list of tuples
        """
        assert args, "must have a group of entities to check against"
        self.INVALID = "INVALID"
        self.intIDs = intIDs
        self.saved = {name: entity for (name, entity) in args}

    def __id2obj__(self, dig):
        return self.intIDs[dig]

    def __wordTranform__(self, x):
        #not implemented
        return [x]

    def __checkIn__(self, entity):
        for k in self.saved:
            if entity in self.saved[k]:
                return entity, k.upper()
        return entity, self.INVALID

    def __call__(self,  ref):
        if ref.isdigit() or "?" in ref:
            return (ref, self.INVALID)
        else:
            r = longestInteger(ref)
            r = self. __id2obj__(r) if r.isdigit() else r

            for w in self.__wordTranform__(r):
                (word, w_type) = self.__checkIn__(w)
                if w_type != self.INVALID:
                    return r, w_type
            return r, self.INVALID

class Accuracy():
    def __init__(round_to=3):
        self.avg_acc = None
        self.n = 0
        self.round_to=round_to

    def __call__(acc):
        self.n += 1
        if self.avg_acc is None:
            self.avg_acc = acc
        else:
            avg_acc = self.avg_acc + ((acc - self.avg_acc) / self.n)
            self.avg_acc = round(avg_acc, self.round_to)

    def getAccuracy():
        return self.avg_acc * 100

class Frequency():
    def __init__(tag_name, round_to=3, it=0, n=0):
        self.tag_name=tag_name
        self.round_to=round_to
        self.it=it
        self.n=n

    def __call__(tag_names):
        #list(set())
        for item in tag_names:
            if self.tag_name in tag_names:
                self.n +=1
            self.it +=1

    def getFrequency():
        return  (self.n/self.it) * 100

# SLICE FUNCTION

def slc2phrase(sent, slc):
    return "".join([s + " " for s in sent[slc]])[:-1]

def slice2firstInd(slc):
    slc = str(slc).lower().replace(" ","").\
            replace("slice(","",).replace(")", "").split(",")
    assert type(slc) is not str
    return int(slc[0])

def slice2lastInd(slc):
    slc = str(slc).lower().replace(" ","").\
            replace("slice(","",).replace(")", "").\
            replace("'", "").replace("none", "").split(",")
    return int(slc[-2])

def closestValidSlice(idx, sent, radius):
    down = (-1 * radius) + idx
    up = radius + idx
    while down < 0 and up > len(sent):
        down += 1 if down < 0 else -1
        up += -1 if up > len(sent) else -1
        if down == up:
            return slice(idx)
    return slice(down, up)

def slice2range(slc):
    f = slice2firstInd(slc)
    e = slice2lastInd(slc)
    return [i for i in range(f, e)]

def foundInds(store: set, slc: slice):
    for i in slice2range(slc):
        store.add(i)

def removeSlc(store: set, slc: slice):
    inds = set()
    foundInds(inds, slc)
    remove(store, list(inds))

def orderSLCinDict(d):
    ks = []
    for k, v in d.items():
        ks.append([k, slice2firstInd(v), v])
    ks = sorted(ks, key=lambda x: x[1])
    return {k: v for k, _, v in ks}, [k for k, _, _ in ks]

