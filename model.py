from jittor import nn, Module

class LSTM(Module):

    def __init__(self, nIn, nHidden, nOut):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def execute(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(Module):

    def __init__(self, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [5, 3, 3, 3] # kernel_size
        ps = [2, 1, 1, 1] # padding
        ss = [1, 1, 1, 1] # stride
        nm = [32, 64, 128, 256] # channel

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU())

        # 1x128x800
        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((4, 2), (4, 2)))  # 32x32x400
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((3, 2), (3, 2)))  # 64x11x200
        convRelu(2, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128x6x100
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 1), (2, 1)))  # 256x2x100

        self.cnn = cnn
        self.rnn = nn.Sequential(
            LSTM(512, nh, nh),    # LSTM (512 -> nh) + Linear (nh -> nh)
            LSTM(nh, nh, nclass)) # LSTM (nh -> nh)  + Linear (nh -> nclass)

    def execute(self, input):
        # conv features
        conv = self.cnn(input)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # [width, batch, feature = channel * height] = [100, batch, 512]
        # rnn features
        output = self.rnn(conv)
        return output

