
import pandas as pd
import numpy as np
import torch as th
from torch import nn,optim
from torch.autograd import Variable as V
from torch.nn import functional as F
import matplotlib.pyplot as plt
from model import CVAE

train = pd.read_csv("dataset/VAE_Train+.csv")
test = pd.read_csv("dataset/VAE_Test+.csv")
trainx, trainy = np.array(train[train.columns[train.columns != "class"]]), np.array(pd.get_dummies(train["class"]))
testx, testy= np.array(test[train.columns[train.columns != "class"]]), np.array(pd.get_dummies(test["class"]))
batch_size = 512
max_epoch = 100
train_N = len(train)
test_N = len(test)
gpu = False
device = "cuda" if gpu else "cpu"


model = CVAE()
if gpu:
    model = model.cuda()
opt = optim.Adadelta(model.parameters(),lr = 1e-3)

def Loss_function(x_hat,x, mu,logsimga):
    reconstraction_loss = F.binary_cross_entropy(x_hat,x,size_average = False)
    KL_div = -0.5 * th.sum(1+logsimga-mu.pow(2) - logsimga.exp())

    return reconstraction_loss+KL_div


def create_batch(x,y):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    y = y[a]
    batch_x = [x[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
    batch_y = [y[batch_size * i : (i+1)*batch_size].tolist() for i in range(len(x)//batch_size)]
    return batch_x, batch_y


def train():
    model.train()
    tr_loss = 0
    batch_x,batch_y = create_batch(trainx,trainy)
    for x,y in zip(batch_x,batch_y):
        opt.zero_grad()
        if gpu:
            x,y = V(th.Tensor(x).cuda()),V(th.Tensor(y).cuda())
        else:
            x,y = V(th.Tensor(x)),V(th.Tensor(y))
        x_hat,mu,logsigma = model(x,y)
        loss = Loss_function(x_hat,x,mu,logsigma)

        loss.backward()
        tr_loss += loss.item()
        opt.step()
    return tr_loss/train_N


def test():
    model.eval()
    te_loss = 0
    batch_x,batch_y = create_batch(testx,testy)
    with th.no_grad():
        for x,y in zip(batch_x,batch_y):
            if gpu:
                x,y = V(th.Tensor(x).cuda()),V(th.Tensor(y).cuda())
            else:
                x,y = V(th.Tensor(x)),V(th.Tensor(y))
            x_hat,mu,sigma = model(x,y)
            loss = Loss_function(x_hat,x,mu,sigma)
            te_loss += loss.item()

    return te_loss/test_N

tr_loss ,te_loss = [],[]
for epoch in range(max_epoch):
    trl = train()
    tel = test()
    tr_loss.append(trl)
    te_loss.append(tel)
    if epoch % 2 == 0:
        print(epoch,trl,tel)

th.save(model.state_dict(), f"save_model/vae_adadelta_{max_epoch}.pth")
plt.plot(tr_loss)
plt.plot(te_loss)
plt.show()