import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class vggNET19(nn.Module):
 
    def __init__(self):

        super(vggNET19, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([

                ('c1',nn.Conv2d(3, 96, kernel_size=3)),
                ('r1',nn.ReLU(inplace=True)),
                ('pool1',nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ('fire1',Fire(96, 16, 64, 64)),
                ('fire2',Fire(128, 16, 64, 64)),
                ('fire3',Fire(128, 32, 128, 128)),
                ('fire4',nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ('fire5',Fire(256, 32, 128, 128)),
                ('fire6',Fire(256, 48, 192, 192)),
                ('fire7',Fire(384, 48, 192, 192)),
                ('fire8',Fire(384, 64, 256, 256)),
                ('pool2',nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ('fire9',Fire(512, 64, 256, 256)),
                ('d1',nn.Dropout(p=0.5)),
                ('c2',nn.Conv2d(512, 10, kernel_size=1)),
                ('r2',nn.ReLU(inplace=True)),
                # ('Apool',nn.AdaptiveAvgPool2d((1, 1)))
        ]))

        self.loss = nn.Sequential(OrderedDict([


            ('sig7', nn.LogSoftmax(dim=-1))
                ]))

    def forward(self, img):
        output = self.convnet(img)
        # print(output.shape)
        output = output.view(img.size(0), -1)
        output = self.loss(output)
        return output
