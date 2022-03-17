import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class cbr2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cbr2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )
    
    def forward(self,x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 1, features = [64,128,256,512]):
        super(UNet, self).__init__()
        self.exps = nn.ModuleList()
        self.cons = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)
        
        #Contracting Path
        for feature in features:
            self.cons.append(cbr2d(in_channel, feature))
            in_channel = feature
            
        #Expansive Path
        for feature in reversed(features):
            self.exps.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2)
            )
            self.exps.append(cbr2d(feature*2, feature))
        
        self.bottleneck = cbr2d(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size= 1)

    def forward(self, x):
        skip_connections = []
        for con in self.cons:
            x = con(x)
            print(x.shape)
            skip_connections.append(x)
            x = self.pool(x)
            print(x.shape)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.exps), 2):
            #up-conv
            print(x.shape)
            x = self.exps[idx](x)
            #get skip_connection
            target = skip_connections[idx//2]
            #if x and target shape doesn't match
            if x.shape != target.shape:
                x = tf.resize(x, size = target.shape[2:])
            #concat target to x
            concat = torch.cat((target, x), dim = 1)
            #next step of up-conv
            print(x.shape)
            x = self.exps[idx+1](concat)
        return self.final_conv(x)

def test():
    x = torch.randn((3,1,160,160))
    model = UNet(in_channel = 1, out_channel = 1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

test()