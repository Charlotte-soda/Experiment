import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from pylab import *  # 支持中文

mpl.rcParams['font.sans-serif'] = ['SimHei']

train_loss = []
train_acc = []


with open("../outdata/train_result.txt", "r") as f:
    Lines = f.readlines()

    count = 0
    # 从文件中loss和acc
    for line in Lines:
        count += 1
        temp = line.split(',')
        train_loss_temp = temp[1]
        train_loss.append(float(train_loss_temp.split(':')[1]))

        train_acc_temp = temp[2]
        train_acc.append(float(train_acc_temp.split(':')[1]))


        print("训练",train_loss_temp.split(':')[1],train_acc_temp.split(':')[1])



iters = range(len(train_loss))
plt.figure()
plt.plot(iters, train_loss, 'r', label='train_loss',linewidth = 2)
plt.plot(iters, train_acc, 'g', label='train_acc',linewidth = 2)
# plt.xlim(xmin = -10, xmax = 205)
# plt.ylim(ymin = 0.1, ymax = 1)
print(plt.axis())
#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}

plt.grid(True)
plt.xlabel('epoch',font2)
plt.ylabel('acc-loss',font2)
plt.legend(loc="lower left")

x_major_locator=MultipleLocator(2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.5)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0,20)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(-0.05,8)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

plt.savefig("../outFigure/" + 'train' + '.jpg')
