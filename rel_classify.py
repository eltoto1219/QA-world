import json

with open("relations.txt", "r") as f:
    data = f.read().split("\n")
    #for r in f:
    #    data.append(r)

def write(f, f_name, j = False):
    if j:
        json.dump(open("{}.json".format(f), "w"), f_name)
    else:
        with open("{}.txt".format(f_name), "w") as file:
            for r in f:
                file.write(r)


purePresentAction = set()
purePastAction = set()
purePositional = set()
positionalPastAction = set()
positionalPresentAction = set()
comparison = set()

#for each category dont forget to weight by frequency
positionalPhrases = {
    'on the other side of',
    'on the front of',
    'on the side of',
    'to the right of',
    'on the back of',
    'to the left of',
    'on the bottom of',
    'on the edge of',
    'in front of',
    'close to',
    'in between',
    'next to',
    'on top of'
    }

pastTenseExceptions = {
    'stuck in',
    'kept in',
    'hung on',
    'worn on',
    'stuck on',
    'sewn on',
    'drawn on',
    'worn around',
    }

complexRelations = {
    'hidden by',
    'full of',
    'seen through',
    'about to hit',
    'hang from',
}

relation2reltype = {}

for r in data:
    if r == '' or r == ' ':
        pass
    split = r.split(" ")
    if "through" in r or r in complexRelations or "to catch" in r or "for" in r or "with" in r or "from" in r:
        complexRelations.add(r)
        relation2reltype[r] = "complex"
    elif "than" in r:
        comparison.add(r)
        relation2reltype[r] = "comparison"
    elif len(split) == 1 and "ing" in r:
        purePresentAction.add(r)
        relation2reltype[r] = "pureAction"
    elif len(split) == 1 or r in positionalPhrases:
        purePositional.add(r)
        relation2reltype[r] = "purePosition"
    elif len(split) > 1 and "ing" in r:
        positionalPresentAction.add(" ".join(split[1:]))
        relation2reltype[r] = "positionalPresentAction"
    elif r in pastTenseExceptions or len(split) > 1 and "ed" in r:
        positionalPastAction.add(" ".join(split[1:]))
        relation2reltype[r] = "positionalPastAction"
    else:
        raise Exception("not accounted for")

write(purePresentAction,"relClassifications/purePresentAction")
write(purePastAction,"relClassifications/purePastAction")
write(purePositional,"relClassifications/purePositional")
write(positionalPastAction,"relClassifications/positionalPastAction")
write(positionalPresentAction,"relClassifications/positionalPresentAction")
write(comparison,"relClassifications/comparison")
write(complexRelations,"relClassifications/complex")
write(relation2reltype,"relClassifications/rel2reltype", j=True)
