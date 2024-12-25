# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from process import process, ProcessImg
import numpy as np
import argparse

# from models import acl_resnet
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


def normalized(x, y, args):
    normalize = args.norm
    print("preprocess_m,n:", x.shape, y.shape)  # torch.Size([1826, 6, 24]) torch.Size([1826, 1])
    # normlized by the maximum value of each row(sensor).
    if (normalize == 2):  # 最大最小归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = MinMaxScaler(min=x[:, :, 0].min(), max=x[:, :, 0].max())
        scaler2 = MinMaxScaler(min=x[:, :, 1].min(), max=x[:, :, 1].max())
        scaler3 = MinMaxScaler(min=x[:, :, 2].min(), max=x[:, :, 2].max())
        scaler4 = MinMaxScaler(min=x[:, :, 3].min(), max=x[:, :, 3].max())
        scaler5 = MinMaxScaler(min=x[:, :, 4].min(), max=x[:, :, 4].max())
        scaler6 = MinMaxScaler(min=x[:, :, 5].min(), max=x[:, :, 5].max())
        scaler_y = MinMaxScaler(min=y.min(), max=y.max())
        x[:, :, 0] = scaler1.transform(x[:, :, 0])
        x[:, :, 1] = scaler2.transform(x[:, :, 1])
        x[:, :, 2] = scaler3.transform(x[:, :, 2])
        x[:, :, 3] = scaler4.transform(x[:, :, 3])
        x[:, :, 4] = scaler5.transform(x[:, :, 4])
        x[:, :, 5] = scaler6.transform(x[:, :, 5])
        y = scaler_y.transform(y)
        print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape,
              y.shape)  # torch.Size([1826, 6, 24]) torch.Size([1826, 1])
        scale.append(scaler1)
        scale.append(scaler2)
        scale.append(scaler3)
        scale.append(scaler4)
        scale.append(scaler5)
        scale.append(scaler6)

    if (normalize == 3):  # 标准差方差归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = StandardScaler(mean=x[:, :, 0].mean(), std=x[:, :, 0].std())
        scaler2 = StandardScaler(mean=x[:, :, 1].mean(), std=x[:, :, 1].std())
        scaler3 = StandardScaler(mean=x[:, :, 2].mean(), std=x[:, :, 2].std())
        scaler4 = StandardScaler(mean=x[:, :, 3].mean(), std=x[:, :, 3].std())
        scaler5 = StandardScaler(mean=x[:, :, 4].mean(), std=x[:, :, 4].std())
        scaler6 = StandardScaler(mean=x[:, :, 5].mean(), std=x[:, :, 5].std())
        scaler_y = StandardScaler(mean=y.mean(), std=y.std())
        x[:, :, 0] = scaler1.transform(x[:, :, 0])
        x[:, :, 1] = scaler2.transform(x[:, :, 1])
        x[:, :, 2] = scaler3.transform(x[:, :, 2])
        x[:, :, 3] = scaler4.transform(x[:, :, 3])
        x[:, :, 4] = scaler5.transform(x[:, :, 4])
        x[:, :, 5] = scaler6.transform(x[:, :, 5])
        y = scaler_y.transform(y)
        print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape, y.shape)
        scale.append(scaler1)
        scale.append(scaler2)
        scale.append(scaler3)
        scale.append(scaler4)
        scale.append(scaler5)
        scale.append(scaler6)
    return x, y, scaler_y


class DataGenerator():
    def __init__(self, args, phase_gen, interval):
        self.args = args
        self.phase_gen = phase_gen
        self.interval = interval
        print(self.phase_gen, self.interval)

        self.speed, self.image, self.label, self.scale_y = self.generator()

    def __len__(self):
        return len(self.label)

    def generator(self):
        print("aaaaaaaaaa")
        list_train = []
        list_val = []
        list_test = []
        im = []
        image = []
        if self.phase_gen == 'train':
            speed, label, scale_y = self.construct_data()
            with open("/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/image_data/path_train.txt", 'r') as f:
                number = 0
                for line in f:
                    if number % self.interval == 0:
                        # print(number)  # 0, 24, 48, 72, .... 43800(43800/24=1825)
                        list_train.append(list(line.rsplit('\n')))
                    number += 1
            total_time_len = len(list_train)
            # print(list_train)
            rang1 = range(1, total_time_len)
            rang2 = range(1, total_time_len - self.args.input_length - self.args.predict_length + 1)
            print("TimeDataset_rang: ", rang1, rang2)  # range(0, 1826) range(0, 1825)
            for i in rang1:
                im.append(ProcessImg(list_train[i][0]))
            print("train convert ok!")
            im = torch.from_numpy(np.array(im))
            print(im.shape)  # torch.Size([1826, 256, 256])
            for j in rang2:
                image.append(im[j:j + self.args.input_length])

            image = torch.stack(image)
            print(speed.shape, image.shape, label.shape)  # torch.Size([1824, 6, 24]) torch.Size([1825, 1, 256, 256]) torch.Size([1824, 1])
            return speed, image, label, scale_y

        elif self.phase_gen == 'val':
            speed, label, scale_y = self.construct_data()
            with open("/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/image_data/path_val(2016).txt", 'r') as f:
                number = 0
                for line in f:
                    if number % self.interval == 0:
                        list_val.append(list(line.rsplit('\n')))
                    number += 1

            total_time_len = len(list_val)
            rang1 = range(total_time_len)
            rang2 = range(total_time_len - self.args.input_length)
            print("TimeDataset_rang: ", rang1, rang2)  # range(0, 8737) range(0, 8713)
            for i in rang1:
                im.append(ProcessImg(list_val[i][0]))

            print("val convert ok!")
            im = torch.from_numpy(np.array(im))
            print(im.shape)  # torch.Size([8737, 256, 256])
            for j in rang2:
                image.append(im[j:j + self.args.input_length])
            image = torch.stack(image)
            print(speed.shape, image.shape, label.shape)     # torch.Size([8713, 24, 256, 256]) torch.Size([8713, 1])
            return speed, image, label, scale_y

        elif self.phase_gen == 'test':
            speed, label, scale_y = self.construct_data()
            with open("/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/image_data/path_test(2017).txt", 'r') as f:
                number = 0
                for line in f:
                    if number % self.interval == 0:
                        list_test.append(list(line.rsplit('\n')))
                    number += 1

            total_time_len = len(list_test)
            rang1 = range(total_time_len)
            rang2 = range(total_time_len - self.args.input_length)
            print("TimeDataset_rang: ", rang1, rang2)  # range(0, 8737) range(0, 8713)
            for i in rang1:
                im.append(ProcessImg(list_test[i][0]))

            print("test convert ok!")
            im = torch.from_numpy(np.array(im))
            print(im.shape)  # torch.Size([8737, 256, 256])
            for j in rang2:
                image.append(im[j:j + self.args.input_length])
            image = torch.stack(image)
            print(speed.shape, image.shape, label.shape)     # torch.Size([8713, 24, 256, 256]) torch.Size([8713, 1])
            return speed, image, label, scale_y

    def construct_data(self):
        if self.phase_gen == 'train':
            f = pd.read_csv(f'/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/Solar-noArea/train.csv', sep=',', index_col=0)
        elif self.phase_gen == 'val':
            f = pd.read_csv(f'/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/Solar-noArea/test.csv', sep=',', index_col=0)
        else:
            f = pd.read_csv(f'/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/Solar-noArea/2017.csv', sep=',', index_col=0)

        feature_file = open(f'/mnt/syanru/CMP-Solar-Wind-peed/Time-24/data/Solar-noArea/list.txt', 'r')
        feature_map = []
        res = []
        for ft in feature_file:
            feature_map.append(ft.strip())

        for feature in feature_map:
            if feature in f.columns:
                res.append(f.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')

        sample_n = len(res[0])
        print("preprocess_sample_n", sample_n, len(res),
              len(res[0]))  # sample_n是所有sample的总数，res是所有特征组合的list，大小为[6x43824] 43824 6 43824

        x_arr, y_arr, data = [], [], []

        res = torch.tensor(res).double()
        node_num, total_time_len = res.shape
        print("TimeDataset_data.shape: ", res.shape)  # [6,43824]

        # rang = range(self.args.input_length, total_time_len - self.args.predict_length + 1, self.interval)
        rang_all = range(0, total_time_len)
        print("rang_all: ", rang_all)  # range(0, 43824)
        rang_ft = range(0, total_time_len - (self.args.input_length + self.args.predict_length) * self.interval, self.interval)
        print("rang_ft:", rang_ft)  # range(0, 43776, 24)
        rang_tar = range((self.args.input_length + self.args.predict_length) * self.interval, total_time_len, self.interval)
        print("rang_tar:", rang_tar)  # range(48, 43824, 24)

        for i in rang_all:
            # print(i)  # 0, 1, 2, 3, .... 43823
            # print(res[:, 0])
            data_1 = res[:, i]
            data.append(data_1)  # 所有数据

        data = torch.stack(data).contiguous()   # 所有数据
        print("all_data.shape:", data, data.shape)  # torch.Size([43823, 6])

        # data = torch.tensor(data).double()
        input_length = self.args.input_length * self.interval
        print(input_length)
        for i in rang_ft:
            ft = data[i: i + input_length, :]

            x_arr.append(ft)

        for j in rang_tar:
            # print("j:", j)  # 48, 72, .... 43800(43800/24=1825)
            tar = data[j, 0]

            y_arr.append(tar)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        y = y.reshape(y.shape[0], 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        print("TimeDataset_x_arr.shape, y_arr.shape, x_arr, y_arr:", x, y, x.shape, y.shape)    # torch.Size([1824, 24, 6]) torch.Size([1824, 1]) # torch.Size([1823, 3, 6]) torch.Size([1823, 1])
        x, y, scale_y = normalized(x, y, self.args)
        print("TimeDataset_normalize_x,y:", x, y, x.shape, y.shape)  # torch.Size([1826, 24, 6]) torch.Size([1826, 1])

        return x, y, scale_y

    def __getitem__(self, idx):
        speed = self.speed[idx].double()
        image = self.image[idx].double()
        label = self.label[idx].double()

        return speed, image, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='batch size', type=int, default=32)
    parser.add_argument('--input_length', help='input_length', type=int, default=1)  # 输入数据长度
    parser.add_argument('--predict_length', help='predict length', type=int, default=1)  # 输出数据长度
    parser.add_argument('--interval', help='EUV数据采样频率（间隔几张采一次）', type=int, default=24)

    args = parser.parse_args()
    main = DataGenerator(args, 'train', args.interval)