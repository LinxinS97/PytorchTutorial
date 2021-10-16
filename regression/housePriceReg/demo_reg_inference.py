import torch
import re
import numpy as np

# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.predict(x)
        return x


# data
ff = open('housing.data').readlines()
data = []
for item in ff:
    # 用正则表达式将2个以上的空格转变为1个
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))

data = np.array(data).astype(np.float)
print(data.shape)
Y = data[:, -1]
X = data[:, 0:-1]

X_train = X[0:496:, ...]
Y_train = Y[0:496:, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

net = Net(13, 1)
net.load_state_dict(torch.load('./model/params_house.pkl'))
loss_func = torch.nn.MSELoss()

# test
x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss = loss_func(pred, y_data) * 0.001
print("loss_test:{}".format(loss))
