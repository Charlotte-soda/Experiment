from cmath import sqrt
from tkinter import font
from wsgiref.validate import InputWrapper

import torch
import torch.nn as nn

from DataProcess.data import getData


def Decision(input1, input2,out1,out2):
  front = (input1 * input2 - out1 * out2)**2
  back = (input1**2 * input2**2) / 2 - (out1**2 * out2**2) / 2
  back = pow(back,2)
  loss = sqrt(front + back)
  return abs(loss)

def GetLoss(input_index, outD):  #参数1：输入数据的下标（80.3），参数2：输出的预测值（80,2）
  indices = torch.tensor([1,2])
  Index = torch.index_select(input_index,1,indices)
  Index = Index.detach().cpu().numpy()

  outData = outD.detach().cpu().numpy()
  Input_data = []
  for item in Index:
    # print("每一个",item)
    eachData = getData(item)
    # print("每一个",eachData)
    Input_data.append(eachData)


  # print("损失函数",Input_data,outData)
  for item in range(len(Input_data)):
      g1 = Input_data[item][0]
      g2 = Input_data[item][1]
      g1_head = outData[item][0]
      g2_head = outData[item][1]
      
      loss = Decision(g1, g2, g1_head, g2_head)
      # print("loss",loss)

  return loss
