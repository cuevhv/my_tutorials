"""x-or usando redes neuronales "planas" en pytorch"""

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
"""entrada-> capa oculta ocultos 4, biases 2"""
"""capa oculta->salida pesos ocultos 2, bias 1 """
print("input->hiden_layer param", params[0].size(), params[1].size())
print("hidden_layer-> output param", params[2].size(), params[3].size())
print(net)

"""
X: varable de entrada, 2 numeros
valores que X puede tener:
[0, 0]
[0, 1]
[1, 0]
[1, 1]
Y: salida, 1 numero:
Salida correspondiente a cada entrada
[0]
[1]
[1]
[0]

"""
y_n = np.array([[0], [1], [1], [0]])
y_n = np.reshape(y_n, [4, 1, 1])
Y = tch.from_numpy(y_n)
Y = Variable(Y.float())
print("Tamano de Y", y_n.shape)


x_n = np.array([[0, 0], [0, 1],[1, 0],[1, 1]])
x_n = np.reshape(x_n, [4, 1, 2])
print("Tamano de X", x_n.shape)
x = tch.from_numpy(x_n)
x = Variable(x.float())

output = net(x)
print("Salida de la red neuronal con pesos aleatoreos y su tamano")
print(output)

"""usamo error cuatrado medio (y-y_hat)^2"""
criterion = nn.MSELoss()

""" El optimizador es gradiente descendiente entonces w = w-lr*df(x)/dw"""
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0)
epoch = 20000

print "Iteramos 20000 veces para minimizar nuestra perdida (loss)"

running_loss = 0.0
for ep in range(0, epoch):
	running_loss = 0.0
	
	optimizer.zero_grad() #funcion propia de pytorch, para que los pesos se retropropagen por cada iteracion y no en conjunto
	output = net(x)
	loss = criterion(output, Y)
	loss.backward()
	optimizer.step()
	running_loss +=loss.data[0]
	if ep % 2000 == 1999:
		print "iteracion", running_loss
	
print(running_loss)
print(tch.round(net(x)))

"""La salida deberia ser un valor similar a nuestro deseado Y"""
