import argparse
import jittor as jt
import jittor
import numpy as np
from jittor import optim, init
from jittor.misc import CTCLoss
import os
import utils
import dataset
from model import CRNN
from time import time
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument('--domain', default='semantic', help='ground truth representations')
parser.add_argument('--modelPath', required=True, help='path to pretrained model')
parser.add_argument('--dataPath', required=True, help='path to dataset')
parser.add_argument('--vocabPath', default='./data', help='path to vocabulary')

opt = parser.parse_args()
print(opt)

domain = opt.domain
model_path = opt.modelPath
batch_size = 16

if jt.has_cuda:
    print("using cuda")
    jt.flags.use_cuda = 1 
else:
    print("using cpu")

converter = utils.strLabelConverter(domain=domain, root=opt.vocabPath)
criterion = CTCLoss(zero_infinity=True)                           

nclass = len(converter.alphabet)
crnn = CRNN(1, nclass, 512)
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(jt.load(model_path))

print('loading dataset from %s' % opt.dataPath)
test_dataset = dataset.lmdbDataset(root=opt.dataPath, 
                                   batch_size=batch_size,
                                   shuffle=True,
                                   transform=dataset.resizeNormalize((800, 128))) 

print('Start test')
crnn.eval()

n_correct = 0
loss_avg = utils.averager()

for batch_idx, (images, raw_texts) in tqdm(enumerate(test_dataset), desc="testing"):
    preds = crnn(images)
    preds = jittor.nn.log_softmax(preds, dim=2)

    text, length = converter.encode(raw_texts)
    loss = criterion(preds, jt.array(text), jt.array([preds.size(0)] * batch_size), jt.array(length)) / batch_size
    loss_avg.add(loss.data)

    preds_index = preds.data.argmax(2)
    preds_index = preds_index.transpose(1, 0)
    sim_preds = converter.decode(preds_index, raw=False)
    for pred, target in zip(sim_preds, raw_texts):
        pred = pred.split()
        target = target.split()
        if pred == target:
            n_correct += 1

raw_preds = converter.decode(preds_index, raw=True)[:10]
for raw_pred, pred, gt in zip(raw_preds, sim_preds, raw_texts):
    print('%-20s \n=> %-20s\ngt: %-20s' % (raw_pred, pred, gt))

accuracy = n_correct / float(len(test_dataset))
print('Test loss: %f, sequence accuracy: %f' % (loss_avg.val(), accuracy))
