import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,Imputer

train = pd.read_csv("dataset/KDDTrain+.txt")
test = pd.read_csv("dataset/KDDTest+.txt")

#since mostly zero 
discard = ["land","logged_in","root_shell","su_attempted","is_host_login","is_guest_login","value"]
for i in discard:
    del train[i]
    del test[i]

# min-max-scaler
numeric_columns = list(train.select_dtypes(include=['int',"float"]).columns)
imp = Imputer(strategy="mean", axis=0)
minmax = MinMaxScaler()
for c in numeric_columns:
    train[c] = minmax.fit_transform(np.array(train[c]).reshape(-1,1))
    test[c] = minmax.fit_transform(np.array(test[c]).reshape(-1,1))

# categry variable transform to onehot
dummy_c = ["protocol_type","flag","service"]
train = pd.get_dummies(train,columns=dummy_c)
test = pd.get_dummies(test,columns=dummy_c)

# add to test columns 
add_f= ['aol', 'harvest', 'http_2784', 'http_8001', 'red_i', 'urh_i']
for i in add_f:
    test["service_"+i] = np.zeros(len(test)).astype("uint8")

# attack grouping and transform to value
dos = ["back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm","mailbomb"]
probe = ["satan","ipsweep","nmap","portsweep","mscan","saint"]
r2l = ["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named","multihop"]
u2r = ["buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"]
attack_categry = {"Dos":dos,"Probe":probe,"R2L":r2l,"U2R":u2r}
a2v = {"normal":0,"Dos":1,"Probe":2,"R2L":3,"U2R":4}

def func(x):
    for c,v in attack_categry.items():
        if x in v:
            return c
    return "normal"
f =lambda x : func(x)
train["class"] = train["class"].map(f)
test["class"] = test["class"].map(f)

train["class"] = train["class"].map(lambda x : a2v[x])
test["class"] = test["class"].map(lambda x : a2v[x])

train.to_csv("dataset/VAE_Train+.csv")
test.to_csv("dataset/VAE_Test+.csv")