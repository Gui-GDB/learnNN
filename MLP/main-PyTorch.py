from itertools import product

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse


class MyDataSet(Dataset):
	"""
	在 MyDataSet类中，存储数据信息以便可以将一批（batch）数据点捆绑在一起（使用 DataLoader），并通过一次前向和反向传播更新权重。
	"""
	def __init__(self, x, y):
		"""
		该方法接受输入和输出，并将它们转换为 torch 浮点对象
		:param x: 输入数据
		:param y: 输出数据
		"""
		# torch.tensor(x).float()
		self.x = x.clone().detach()
		self.y = y.clone().detach()

	def __getitem__(self, index):
		"""
		获取数据样本，
		:param index: 为从数据集中获取到的索引
		:return: 返回获取到数据样本
		"""
		return self.x[index], self.y[index]

	def __len__(self):
		"""
		指定数据长度
		:return: 返回输入数据的长度
		"""
		return len(self.x)


class MyNeuraNet(torch.nn.Module):
	"""
	创建神经网络
	继承自 nn.Module, nn.Module 是所有神经网络模块的基类
	"""
	def __init__(self, inputs, outputs, hidden):
		"""
		使用 __init__ 方法初始化神经网络的所有组件
		调用 super().__init__() 确保类继承 nn.Module，可以利用 nn.Module 编写的所有预构建函数。
		"""
		super().__init__()
		# 全连接层(包含了偏置参数)
		self.input_to_hidden_layer = nn.Linear(in_features=inputs, out_features=hidden)
		# 使用 ReLU 激活函数
		# self.hidden_layer_activation = nn.ReLU()
		# 使用 Sigmod 激活函数
		self.hidden_layer_activation = nn.Sigmoid()
		# 全连接层(包含了偏置参数)
		self.hidden_to_output_layer = nn.Linear(in_features=hidden, out_features=outputs)

	def forward(self, x):
		"""
		将初始化后的神经网络组件连接在一起，并定义网络的前向传播方法 forward
		必须使用 forward 作为前向传播的函数名，因为 PyTorch 保留此函数作为执行前向传播的方法，使用其他名称会引发错误。
		:param x: 输入数据
		:return: 返回前向传播预测的结果
		"""
		# 等价于 x @ self.input_to_hidden_layer
		x = self.input_to_hidden_layer(x)
		x = self.hidden_layer_activation(x)
		x = self.hidden_to_output_layer(x)
		return x


def train(epoch, dataLoader, model):
	# 定义损失函数，由于需要预测连续变量，因此使用均方误差作为损失函数：
	loss_fn = nn.MSELoss()
	# 定义用于降低损失值的优化器，优化器的输入是与神经网络相对应的参数（权重与偏置）以及更新权重时的学习率。
	optimizer = Adam(myNet.parameters(), lr=args.lr)

	# 保存每次迭代的损失值
	loss_values = []
	model.train()
	for _ in range(1, epoch + 1):
		loss = 0.0
		for batch in dataLoader:
			x, y = batch
			# 梯度清零
			optimizer.zero_grad()
			# 计算损失值
			loss_value = loss_fn(model(x), y)
			# 梯度下降
			loss_value.backward()
			# 更新权重
			optimizer.step()
			loss += loss_value.item()

		loss_values.append(loss / len(dataLoader))
	return loss_values


if __name__ == '__main__':
	parser = argparse.ArgumentParser("PyTorch 构建简单的神经网络")
	parser.add_argument('-l', '--lr', type=float, default=0.005, help='learning rate')
	parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs')
	parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size')
	parser.add_argument('-s', '--seed', type=int, default=2003, help='random seed')
	args = parser.parse_args()

	# 初始化数据集，定义输入（x）和输出（y）,输入中的每个列表的值之和就是输出列表中对应的值
	inputs = np.array([[1, 1], [2, 1], [6, 4], [6, 9],
					   [5, 3], [2, 6], [1, 5], [9, 7],
					   [8, 3], [4, 5], [3, 9], [7, 7],
					   [6, 5], [7, 2], [1, 3], [2, 3],
					   [8, 9], [9, 9]])
	# 标准输出结果
	outputs = np.array([[2], [3], [10], [15],
						[8], [8], [6], [16],
						[11], [9], [12], [14],
						[11], [9], [4], [5],
						[17], [18]])
	# 将输入列表对象转换为张量对象
	x = torch.tensor(inputs, dtype=torch.float)
	y = torch.tensor(outputs, dtype=torch.float)
	# 获取当前是否有可以使用的 cuda 设备
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# 将输入数据和输出数据点注册到 device 中。
	x = x.to(device)
	y = y.to(device)

	# 指定随机种子，保证每次神经网络都使用的是相同的随机值，利于代码复现
	print('using device: ' + torch.cuda.get_device_name(0))
	torch.manual_seed(args.seed)
	if device == 'cuda':
		torch.cuda.manual_seed(args.seed)

	# 创建 MyNeuralNet 类对象的实例并将其注册到 device
	myNet = MyNeuraNet(2, 1, 10).to(device)

	# 实例化数据集
	myDataset = MyDataSet(x, y)
	# 通过 DataLoader 传递数据实例，从原始输入输出对象中获取 batch_size 个数据点
	dataLoader = DataLoader(dataset=myDataset, batch_size=args.batch_size, shuffle=True)

	# 模型开始训练
	loss_values = train(args.epochs, dataLoader, myNet)

	# 利用训练的模型预测
	test_data = np.array(list(product(range(1, 10), repeat=2)))
	# 标准输出结果
	test_ans = np.array([[x + y] for x, y in test_data])
	# 模型预测
	test_outputs = myNet(torch.tensor(test_data, dtype=torch.float).to(device))
	# 结果四舍五入，不保留小数
	test_outputs_approximate = np.rint(test_outputs.detach().cpu().numpy()).astype(int)
	# 打印结果
	count = 0.0
	for input_data, approximate, output_data, stand_ans in zip(test_data, test_outputs_approximate, test_outputs, test_ans):
		input_data = list(map(int, input_data))  # 转成普通 int
		print(f"{input_data} -> {approximate[0]} ({output_data[0]})")
		count += 1 if approximate[0] == stand_ans[0] else 0
	print(f"模型预测的准确率为：{count / len(test_ans):.4f}")

	# 绘制损失函数
	plt.plot(loss_values)
	plt.title("Loss variation over increasing epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Loss value")
	plt.show()
