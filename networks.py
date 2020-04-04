import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

class EfficientNetExtractor(nn.Module):
    def __init__(self, version):
        super(EfficientNetExtractor, self).__init__()
        assert version in [f'b{id}' for id in range(8)]
        self.extractor = EfficientNet.from_pretrained(f'efficientnet-{version}')
        
    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size()[0], -1)
        x = self.extractor._dropout(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
