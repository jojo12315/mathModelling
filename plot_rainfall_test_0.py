import math
import os
import shutil
from glob import glob

import numpy
import torch
import torch.nn as nn

from rainfall_dataUtil import MyData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_path = "rainfal_SampleData\\"
    train_data = MyData(train_path, "test")
    all_label = torch.Tensor([item.detach().numpy() for item in train_data.labels])
    train_label = all_label
    label = train_label[0]
    att_Unet_test = torch.unsqueeze(torch.Tensor(numpy.load(glob(os.path.join("test_rainfall_pred_att","*"))[0])),dim=0)
    Unet_test = torch.unsqueeze(torch.Tensor(numpy.load(glob(os.path.join("test_rainfall_pred","*"))[0])),dim=0)
########################################################绘图##################################################################################################################
    # plt.figure(figsize=(27, 9))
    #
    #
    #
    #
    # ax1 = plt.subplot(1, 3, 1)
    # plt.imshow(torch.squeeze(label,dim=0))
    #
    #
    #
    # ax2 = plt.subplot(1, 3, 2)
    # plt.imshow(torch.squeeze(Unet_test,dim=0))
    #
    #
    #
    # ax3 = plt.subplot(1, 3, 3)
    # plt.imshow(torch.squeeze(att_Unet_test,dim=0))
    #
    # plt.savefig("pred_rainfall_test_0.png")
    # plt.show()
####################################################################################################################################################################################
    c1 = nn.L1Loss()
    c2 = nn.MSELoss()
    print("Unet:")
    print("MAE:{:.5f}".format(c1(Unet_test, label)))
    print("MSE:{:.5f}".format(c2(Unet_test, label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(Unet_test, label))))
    print("att_Unet:")
    print("MAE:{:.5f}".format(c1(att_Unet_test, label)))
    print("MSE:{:.5f}".format(c2(att_Unet_test, label)))
    print("RMSE:{:.5f}".format(torch.sqrt(c2(att_Unet_test, label))))








