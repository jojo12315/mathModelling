from model import ConvLSTM
import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, int(in_planes // ratio), (1,1),stride=1,padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(int(in_planes // ratio), in_planes, (1,1),stride=1,padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class AttentionBasedConvLSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, kernel_size, num_layers,batch_first=True, bias=True, return_all_layers=False):
        super(AttentionBasedConvLSTM, self).__init__()
        self.att = Attention(3)
        self.convL = ConvLSTM(input_dim,hidden_dim,kernel_size,num_layers,batch_first,bias)

    def forward(self,inputs):
        att = torch.zeros(inputs.shape,device=inputs.device)
        for i in range(inputs.shape[1]):
            att[:,i,:,:,:] = self.att(inputs[:,i,:,:,:])
        att_inputs = inputs.mul(att)
        out = self.convL(att_inputs)
        return out




