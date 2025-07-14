import random
from models.layers import *



class VGGSNN(nn.Module):
    def __init__(self, t):
        super(VGGSNN, self).__init__()
        self.t = t
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(self.t,2,64,3,1,1),
            Layer(self.t,64,128,3,1,1),
            pool,
            Layer(self.t,128,256,3,1,1),
            Layer(self.t,256,256,3,1,1),
            pool,
            Layer(self.t,256,512,3,1,1),
            Layer(self.t,512,512,3,1,1),
            pool,
            Layer(self.t,512,512,3,1,1),
            Layer(self.t,512,512,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        self.classifier = SeqToANNContainer(WrapedSNNOp(nn.Linear(512*W*W,10), self.t))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = input.to(torch.float32)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x
    