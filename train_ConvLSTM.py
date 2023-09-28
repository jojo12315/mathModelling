from dataUtil import MyData
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from model import *
import matplotlib.pyplot as plt

# class ConvL(nn.Module):
#     def __init__(self,input_dim, hidden_dim, kernel_size, num_layers,batch_first=False, bias=True, return_all_layers=False):
#         super(ConvL, self).__init__()
#         self.Conv_LSTM = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,batch_first=False, bias=True, return_all_layers=False)


if __name__ == '__main__':
    device = "cuda:0"
    train_path = "SampleData\\"

    train_data= MyData(train_path)

    train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=False)

    epoch = 100
    model = ConvLSTM(3, 1 ,(3,3), 1, True, False, False)
    model = model.to(device)
    c = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    auc_sum = []
    for i in range(epoch):
        all_result = []
        all_label = []
        l = 0
        for dbz,kdp,zdr,label in train_dataloader:
            dbz = torch.unsqueeze(dbz, dim=4)
            kdp = torch.unsqueeze(kdp, dim=4)
            zdr = torch.unsqueeze(zdr, dim=4)
            train_input = torch.cat([dbz,kdp,zdr],dim=4)
            train_label = torch.unsqueeze(label, dim=4)

            train_input = train_input.permute(0, 1, 4, 3, 2)
            train_label = train_label.permute(0, 1, 4, 3, 2)

            train_input = train_input.to(device)
            train_label = train_label.to(device)
            result = model(train_input)
            pred = result[0][0]
            loss = c(pred, train_label)
            l += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch[{}/{},Loss:{:.5f}]".format(i + 1, epoch, l/len(train_dataloader)))
    torch.save(model.state_dict(), "Conv_LSTM.pt")





