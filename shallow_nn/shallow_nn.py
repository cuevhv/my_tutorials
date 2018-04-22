"""x-or using a shallow-nn in pytorch"""

import numpy as np
import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class shallow_nn(nn.Module):
	def __init__(self):
		super(shallow_nn, self).__init__()
		self.fc1 = nn.Linear(2, 2)
		self.fc2 = nn.Linear(2, 1)
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x

	
net = shallow_nn()
params = list(net.parameters())
print("size of parameters weights", len(params))
"""input-> hidden 4 weights and 2 biases"""
"""hidden->output 2 weights and 1 bias """
print("input->hiden_layer param", params[0].size(), params[1].size())
print("hidden_layer-> output param", params[2].size(), params[3].size())
print(net)

"""
X: input variable, 2 numbers
possible values:
[0, 0]
[0, 1]
[1, 0]
[1, 1]
Y: output, 1 number
outputs of the inputs:
[0]
[1]
[1]
[0]

"""
y_n = np.array([[0], [1], [1], [0]])
y_n = np.reshape(y_n, [4, 1, 1])
Y = tch.from_numpy(y_n)
Y = Variable(Y.float())
print("y_shape", y_n.shape)


x_n = np.array([[0, 0], [0, 1],[1, 0],[1, 1]])
x_n = np.reshape(x_n, [4, 1, 2])
print("x_shape", x_n.shape)
x = tch.from_numpy(x_n)
x = Variable(x.float())

output = net(x)
print("output with random weights and its size")
print(output)

"""we use MSE (y-y_hat)**2"""
criterion = nn.MSELoss()

""" The optimizer is an stochastic gradient descent so w = w-df/dw"""
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0)
epoch = 20000

print "We iterate 10000 times to reduce our loss"

running_loss = 0.0
for ep in range(0, epoch):
	running_loss = 0.0
	
	optimizer.zero_grad() #function proper of pytorch, so it only backpropagates the weights of the current epoch
	output = net(x)
	loss = criterion(output, Y)
	loss.backward()
	optimizer.step()
	running_loss +=loss.data[0]
	if ep % 2000 == 1999:
		print "epoch", running_loss
	
print(running_loss)
print(tch.round(net(x)))

"""should output a value equal to Y"""
