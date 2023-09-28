import math
import os
import shutil
from glob import glob

def getIndex(num):
    if num < 42:
        return 0
    elif num < 2*42:
        return 1
    elif num < 3*42:
        return 2
    elif num < 4*42:
        return 3
    elif num < 5*42:
        return 4
    # elif num < 6*21:
    #     return 5
    # elif num < 7*21:
    #     return 6
    # elif num < 8*21:
    #     return 7
    # elif num < 9*21:
    #     return 8
    # elif num < 10*21:
    #     return 9

if __name__ == '__main__':
    path = "D:/BaiduNetdiskDownload/NJU_CPOL_update2308/NJU_CPOL_update2308/"
    savePath = "SampleData"
    dataClass = ["dBZ","KDP","ZDR"]
    height = ["3.0km"]
    dataSet = ["train","validate","test"]
    for c in dataClass:
        for h in height:
            count = 0
            dirs = os.listdir(path+c+"/"+h+"/")
            dir_num = len(dirs)
            train_num = math.floor(dir_num*0.6)
            validate_num = math.floor(dir_num * 0.8)
            test_num = dir_num
            lenCount = [0 for z in range(10)]
            for i in range(0,train_num):
                files = os.listdir(path+c+"/"+h+"/"+dirs[i]+"/")
                file_num = len(files)
                if lenCount[getIndex(file_num)] > 1:
                    continue
                lenCount[getIndex(file_num)] += 1
                sample_num = file_num - 20 + 1
                for j in range(sample_num):
                    os.makedirs(savePath+"/"+c+"/"+h+"/"+dataSet[0]+"/sample"+str(count))
                    for k in range(20):
                        shutil.copy(path+c+"/"+h+"/"+dirs[i]+"/"+files[j+k], savePath+"/"+c+"/"+h+"/"+dataSet[0]+"/sample"+str(count) +"/"+files[j+k])
                    count+=1
            count = 0
            lenCount = [0 for z in range(10)]
            for i in range(train_num,validate_num):
                files = os.listdir(path+c+"/"+h+"/"+dirs[i]+"/")
                file_num = len(files)
                if lenCount[getIndex(file_num)] > 1:
                    continue
                lenCount[getIndex(file_num)] += 1
                sample_num = file_num - 20 + 1
                for j in range(sample_num):
                    os.makedirs(savePath+"/"+c+"/"+h+"/"+dataSet[1]+"/sample"+str(count))
                    for k in range(20):
                        shutil.copy(path+c+"/"+h+"/"+dirs[i]+"/"+files[j+k], savePath+"/"+c+"/"+h+"/"+dataSet[1]+"/sample"+str(count) +"/"+files[j+k])
                    count+=1
            count = 0
            lenCount = [0 for z in range(10)]
            for i in range(validate_num,test_num):
                files = os.listdir(path+c+"/"+h+"/"+dirs[i]+"/")
                file_num = len(files)
                if lenCount[getIndex(file_num)] > 1:
                    continue
                lenCount[getIndex(file_num)] += 1
                sample_num = file_num - 20 + 1
                for j in range(sample_num):
                    os.makedirs(savePath+"/"+c+"/"+h+"/"+dataSet[2]+"/sample"+str(count))
                    for k in range(20):
                        shutil.copy(path+c+"/"+h+"/"+dirs[i]+"/"+files[j+k], savePath+"/"+c+"/"+h+"/"+dataSet[2]+"/sample"+str(count) +"/"+files[j+k])
                    count+=1


