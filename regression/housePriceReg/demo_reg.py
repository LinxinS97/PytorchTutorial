# housing.data: 前13列为属性值，最后一列为房价
import torch

# data
import re
import numpy as np

# 读取房价数据集中的所有行
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


# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    # 我们需要保证y_data和pred的维度是一致的，这里y_data的维度是[496, 1], pred的维度是[496]，因此我们可以调用squeeze把这个多余的1删除
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("ite:{}, loss_train:{}".format(i, loss))
    print(pred[0:10])
    print(y_data[0:10])

    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001
    print("ite:{}, loss_test:{}".format(i, loss))

# 保存整个模型（可以直接load）
# torch.save(net, "model/model.pkl")
# torch.load()
# 只保存参数（需要重新定义网络）
torch.save(net.state_dict(), "model/params_house.pkl")
# net.load_state_dict(torch.load('...'))
