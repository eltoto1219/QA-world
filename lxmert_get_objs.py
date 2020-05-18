import utils
import os
import sys



"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""

gqa_buffer_loader = GQABufferLoader()

        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:
            # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)


if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


Split2ImgFeatPath = {
    'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label

        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("data/lxmert/%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),




### gqa pretaining data item

datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        return example


# data processing on img dataset

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)
        print("Use %d data in torch dataset" % (len(self.data)))


### load obj features from tsv

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

