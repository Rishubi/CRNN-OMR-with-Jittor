#!/usr/bin/python
# encoding: utf-8

import jittor as jt
from jittor.dataset.dataset import Dataset
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm

class lmdbDataset(Dataset):

    def __init__(self, root, transform=None, batch_size=1, shuffle=False, num_workers=0):
        super(lmdbDataset, self).__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.transform = transform
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.total_len = int(txn.get('num-samples'.encode()))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, num_workers=num_workers)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

        return (img, label)


class OMRDataset(Dataset):
    def __init__(self, root="data", mode="train", domain="semantic", distorted=False, transform=None, batch_size=1, shuffle=False, num_workers=0):
        super(OMRDataset, self).__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.domain = domain
        self.distorted = distorted

        paths = []
        labels = []
        print("reading paths...")
        path_saved = os.path.join(root, "paths_{}{}".format(domain, "_distorted.json" if distorted else ".json"))
        label_saved = os.path.join(root, "labels_{}{}".format(domain, "_distorted.json" if distorted else ".json"))
        if os.path.exists(path_saved) and os.path.exists(label_saved):
            paths = json.loads(open(path_saved, 'r').read())
            labels = json.loads(open(label_saved, 'r').read())
        else:
            items_path = os.path.join(root, "{}.txt".format(mode))  # 需要放入数据集的数据项名称列表所在文件，可自行设置
            with open(items_path, 'r') as fr:
                for line in tqdm(fr.readlines(), desc="{} data".format(mode)):
                    line = line.strip()
                    path = os.path.join(root, 'Corpus', line, line + ('_distorted.jpg' if distorted else '.png'))
                    paths.append(path)
                    with open(os.path.join(root, 'Corpus', line, "{}.{}".format(line, domain)), 'r') as f:
                        label = f.read().strip().split()
                        labels.append(label)
            with open(path_saved, 'w') as f:
                f.write(json.dumps(paths))
            with open(label_saved, 'w') as f:
                f.write(json.dumps(labels))
        print("paths read")
        self.paths = paths
        self.labels = labels
        self.total_len = len(self.paths)

        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, num_workers=num_workers)
        print("{} dataset loaded".format(mode))

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            img = Image.open(path).convert('L')  # grey-scale
            img = self.transform(img)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        return (img, self.labels[index])


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = (np.array(img) / 127.5) - 1.0
        return np.expand_dims(img, axis=0).astype(np.float32)
