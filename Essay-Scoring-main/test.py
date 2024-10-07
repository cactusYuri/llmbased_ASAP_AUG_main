import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
from metrics import *
from utils import rescale_tointscore
from utils import domain_specific_rescale
import data_prepare
from hierarchical_att_model import HierAttNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
from reader import *



logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--test', type=str, default='test.txt',
                        help='test file')
    args = parser.parse_args()

    # 加载保存的模型
    model = torch.load('net.pkl')

    # 假设新的数据为 x_new，需要将其转换为适合模型输入的格式，例如张量
    x_new = torch.tensor(...)  # 这里需要根据实际数据进行转换

    # 如果模型在训练时使用了 GPU，并且当前环境有 GPU，将数据和模型移到 GPU
    if torch.cuda.is_available():
        model.cuda()
        x_new = x_new.cuda()

    # 进行预测
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        predictions = model(x_new)

    # 对预测结果进行后续处理
    print(predictions)



if __name__ == '__main__':
    main()
