from __future__ import print_function
from __future__ import division

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

parser.add_argument('--expr_dir', default='runs', help='Where to store models')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--root', default='data', help='path to dataset directory')
parser.add_argument('--trainRoot', required=True, help='path to train dataset')
parser.add_argument('--valRoot', required=True, help='path to validation dataset')
parser.add_argument('--domain', default='semantic', help='ground truth representations')
parser.add_argument('--distort', action='store_true', help='use distorted images or not')
parser.add_argument('--nh', type=int, default=512, help='size of the lstm hidden state')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--num_workers', type=int, default=8, help="")

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, not used by adadelta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is adam)')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop (default is adam)')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')

parser.add_argument('--n_val_disp', type=int, default=10, help='Number of samples to display when val')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=400, help='Interval to val')
parser.add_argument('--saveInterval', type=int, default=400, help='Interval to save')

opt = parser.parse_args()
print(opt)

if jt.has_cuda:
    print("using cuda")
    jt.flags.use_cuda = 1
else:
    print("using cpu")

root = opt.root
batch_size = opt.batchSize
domain = opt.domain
distorted = opt.distort
mode = "distorted" if opt.distort else "norm"
expr_dir = os.path.join(opt.expr_dir, domain, mode)
if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

print("Loading datasets...")
train_dataset = dataset.lmdbDataset(root=opt.trainRoot, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers = opt.num_workers,
                                    transform=dataset.resizeNormalize((800, 128)))

val_dataset = dataset.lmdbDataset(root=opt.valRoot, 
                                   batch_size=batch_size,
                                   shuffle=True,
                                   transform=dataset.resizeNormalize((800, 128))) 

assert train_dataset

converter = utils.strLabelConverter(domain=domain, root=root)
criterion = CTCLoss(zero_infinity=True)                           

nclass = len(converter.alphabet)
nc = 1
crnn = CRNN(nc, nclass, opt.nh)
print(crnn)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(jt.load(opt.pretrained))
else:
    crnn.apply(weights_init)


# setup optimizer
if opt.rmsprop:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
elif opt.adadelta:
    print("Jittor doesn't support Adadelta now.")
    exit(0)
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def val(max_iter=20):
    print('Start val')
    crnn.eval()

    n_correct = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(val_dataset))

    for batch_idx, (images, raw_texts) in tqdm(enumerate(val_dataset), desc="validating"):
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
        if batch_idx >= max_iter:
            break

    raw_preds = converter.decode(preds_index, raw=True)[:opt.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, raw_texts):
        print('%-20s \n=> %-20s\ngt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * batch_size)
    print('Val loss: %f, sequence accuracy: %f' % (loss_avg.val(), accuracy))


def train():
    print('Start train')
    crnn.train()
    loss_avg = utils.averager()
    t0 = time()
    for batch_idx, (images, raw_texts) in enumerate(train_dataset):
        i = batch_idx + 1
        preds = crnn(images)
        preds = jittor.nn.log_softmax(preds, dim=2)
        text, length = converter.encode(raw_texts)
        loss = criterion(preds, jt.array(text), jt.array([preds.size(0)] * batch_size), jt.array(length)) / batch_size
        
        optimizer.step(loss)
        loss_avg.add(loss.data)

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f  Time per batch: %f' %
                  (epoch, opt.nepoch, i, int(len(train_dataset) / batch_size), loss_avg.val(), (time() - t0) / opt.displayInterval))
            loss_avg.reset()
            t0 = time()

        if i % opt.valInterval == 0:
            val()
            crnn.train()

        if i % opt.saveInterval == 0:
            jt.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pkl'.format(expr_dir, epoch, i))

for epoch in range(1, opt.nepoch + 1):
    train()
