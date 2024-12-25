# coding:utf-8
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
list_train=[]
list_val=[]
# with open("/home/syanru/Solar-Wind-peed/CNN-code/data/path_test(2017).txt", 'r') as f:
# with open("/home/syanru/Solar-Wind-peed/CNN-code/data/path_val(2016).txt", 'r') as f:
with open("/home/syanru/Solar-Wind-peed/CNN-code/data/path_train.txt", 'r') as f:
    for line in f:
        list_train.append(list(line.rsplit('\n')))

print(len(list_train))
result = []
for i in range(0, len(list_train)):
    temp = 0
    print(list_train[i][0])
    img = Image.open(os.path.join('/home/syanru/Solar-Wind-peed/',list_train[i][0]))
    img_array = img.load()
    width = img.size[0]
    height = img.size[1]
    # print("width = ", width)
    # print("height = ", height)
    for h in range(0, height):
        for w in range(0, width):
            pixel = img_array[w, h]
            temp += pixel
            # print(' ', pixel, end="")
        # print('\n')
    result.append(temp)
    # print(result[i])
print(result)
print(type(result[0]))
'''
file_handle = open('/home/sunyanru19s/solar_wind_coding/LSTM/data/area_test(2017).txt', mode='a+')
for i in range (0,len(result)):
    file_handle.write(str(result[i]))
    file_handle.write('\n')
'''
#with open("/home/sunyanru19s/solar_wind_coding/LSTM/data/area_val(2016).txt","wb") as f:
#    for list_mem in result:
#        f.write(list_mem)
print("write OKKKKKKK!!!!!")