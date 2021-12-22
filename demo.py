import argparse
import jittor as jt
import numpy as np
import utils
import dataset
from PIL import Image
from model import CRNN
parser = argparse.ArgumentParser()

parser.add_argument('--modelPath', required=True, help='path to pretrained model')
parser.add_argument('--imagePath', required=True, help='path to image to be inferenced')
parser.add_argument('--vocabPath', default='./data', help='path to vocabulary')

opt = parser.parse_args()
print(opt)

if jt.has_cuda:
    print("using cuda")
    jt.flags.use_cuda = 1 
else:
    print("using cpu")


model_path = opt.modelPath
img_path = opt.imagePath

converter = utils.strLabelConverter(domain='semantic', root=opt.vocabPath)

transformer = dataset.resizeNormalize((800, 128))
image = Image.open(img_path).convert('L')
image = transformer(image)
image = jt.array(np.expand_dims(image, axis=0))

crnn = CRNN(1, len(converter.alphabet), 512)
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(jt.load(model_path))

crnn.eval()
preds = crnn(image)
preds = jt.nn.log_softmax(preds, dim=2)
preds_index = preds.data.argmax(2)
preds_index = preds_index.transpose(1, 0)

raw_pred = converter.decode(preds_index, raw=True)
sim_pred = converter.decode(preds_index, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
