
import matplotlib.pyplot as plt
import pylab
import torch as th
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report
from model import CVAE
import numpy as np
def Loss_function(x_hat,x, mu,logsimga):
    reconstraction_loss = F.binary_cross_entropy(x_hat,x,size_average = False)
    KL_div = -0.5 * th.sum(1+logsimga-mu.pow(2) - logsimga.exp())

    return reconstraction_loss+KL_div


model = CVAE()
param = th.load('save_model/vae_adadelta300.pth',map_location=lambda x,y:x)
model.load_state_dict(param)

test = pd.read_csv("dataset/VAE_Test+.csv")
testx, testy= np.array(test[test.columns[test.columns != "class"]]), np.array(pd.get_dummies(test["class"]))
z_dim = 25
n,m = testx.shape[1],testy.shape[1]
test_label = th.eye(m)
attack_name = ["normal","Dos","Probe","R2L","U2R"]
pred = []
for x in testx:
    each_loss = []
    x = th.Tensor(x.reshape(1,n))
    for label in test_label:
        label = th.Tensor(label.reshape(1,m))
        x_hat,mu,sigma = model(x,label)
        loss = Loss_function(x_hat,x,mu,sigma)
        each_loss.append(loss)
    pred.append(np.identity(m)[np.argmin(each_loss)])

print(classification_report(testy,np.array(pred),target_names=attack_name))

# visualize latent space 
if z_dim == 2:
    z = model.encoder(V(th.Tensor(trainx)))
    mu,var = z
    mu,var = mu.detach().numpy(),var.detach().numpy()
    plt.figure(figsize=(10,10))
    plt.scatter(mu[:,0], mu[:,1],marker =".", c = label, cmap=pylab.cm.jet,alpha=0.4)
    plt.colorbar()
    plt.show()