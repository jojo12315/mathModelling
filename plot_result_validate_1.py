import math
import os
import shutil
from glob import glob

import numpy
import torch
import torch.nn as nn

from dataUtil import MyData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_path = "SampleData\\"
    train_data = MyData(train_path, "validate")
    all_label = torch.Tensor([item.detach().numpy() for item in train_data.dbz_labels])
    train_label = torch.unsqueeze(all_label, dim=4)
    train_label = train_label.permute(0, 1, 4, 3, 2)


    att_ConvLSTM_test = torch.squeeze(torch.Tensor(numpy.load(glob(os.path.join("att_ConvLSTM_pred_validate","*"))[1])),dim=0)

    Conv_LSTM_test = torch.squeeze(torch.Tensor(numpy.load(glob(os.path.join("ConvLSTM_pred_validate","*"))[1])),dim=0)

    Unet3D_test = torch.squeeze(torch.Tensor(numpy.load(glob(os.path.join("Unet3D_pred","*"))[1])).permute(0, 2, 1, 3, 4),dim=0)

    # att_ConvLSTM_validate = glob(os.path.join("att_ConvLSTM_pred_validate", "*"))
    # plt.figure(figsize=(36, 90))

    label = train_label[1]


    # for i in range(label.shape[0]):
    #     ax = plt.subplot(10, 4, (i + 1)*3+i-2)
    #     plt.imshow(torch.squeeze(label[i],dim=0))
    #
    #
    # for i in range(label.shape[0]):
    #     ax = plt.subplot(10, 4, (i + 1)*3+i-1)
    #     plt.imshow(torch.squeeze(Unet3D_test[i],dim=0))
    #
    #
    # for i in range(label.shape[0]):
    #     ax = plt.subplot(10, 4, (i + 1)*3+i)
    #     plt.imshow(torch.squeeze(Conv_LSTM_test[i],dim=0))
    #
    #
    # for i in range(label.shape[0]):
    #     ax = plt.subplot(10, 4, (i + 1)*3+i+1)
    #     plt.imshow(torch.squeeze(att_ConvLSTM_test[i],dim=0))
    # plt.savefig("pred_result_validate_1.png")
    # plt.show()

    c1 = nn.L1Loss()
    c2 = nn.MSELoss()
    print("Unet:")
    print("MAE:{:.5f}".format(c1(Unet3D_test, label)))
    print("MSE:{:.5f}".format(c2(Unet3D_test, label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(Unet3D_test, label))))
    print("ConvLSTM:")
    print("MAE:{:.5f}".format(c1(Conv_LSTM_test, label)))
    print("MSE:{:.5f}".format(c2(Conv_LSTM_test, label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(Conv_LSTM_test, label))))
    print("att_ConvLSTM:")
    print("MAE:{:.5f}".format(c1(att_ConvLSTM_test, label)))
    print("MSE:{:.5f}".format(c2(att_ConvLSTM_test, label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(att_ConvLSTM_test, label))))










