from itertools import product
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import argparse


def feed_forward(inputs, outputs, weights):
	"""
	前向传播的过程
	:param inputs: 输入数据
	:param outputs: 输出数据的正确值
	:param weights: 神经网络中的权重参数和偏置项
	:return: 损失值
	"""
	# 隐藏层1
	pre_hidden = np.dot(inputs, weights[0]) + weights[1]
	# Sigmod 非线性激活函数
	hidden = 1 / (1 + np.exp(-pre_hidden))
	# 输出层
	pred_output = np.dot(hidden, weights[2]) + weights[3]
	# Sigmod 非线性激活函数
	# pred_output = 1 / (1 + np.exp(-pred_output))
	# 计算连续变量的损失值（loss）
	mean_squared_error = np.mean(np.square(pred_output - outputs))

	return mean_squared_error


def update_weights(inputs, outputs, weights, lr):
	"""
	反向传播更新权重参数（不是标准的反向传播）
	:param inputs: 输入数据
	:param outputs: 输出的标准结果
	:param weights: 权重参数
	:param lr: 学习率（超参数）
	:return: 返回更新后的权重参数与当前权重参数的损失值
	"""
	# 定义每个权重参数增加的一个非常小的量(注意：这个不是 weight_decay)，数值查分里的步长。
	h = 1e-6
	# 计算当前权重下的损失值
	original_loss = feed_forward(inputs, outputs, weights)
	# 创建一个原始权重参数的副本用来保存更新后的权重参数
	updated_weights = deepcopy(weights)
	# 遍历每一个参数进行更新
	for i, layer in enumerate(weights):
		for index, weight in np.ndenumerate(layer):
			temp_weights = deepcopy(weights)
			# 为神经网络中的每一个参数增加一个小的值
			temp_weights[i][index] += h
			# 计算更新一个权重后的损失值(每个参数都要去算一遍损失值，效率十分的低下，可以使用链式法则计算梯度下降)
			loss_plus = feed_forward(inputs, outputs, temp_weights)
			# 计算梯度
			grad = (loss_plus - original_loss) / h
			# 更新当前权重参数
			updated_weights[i][index] -= lr * grad
	return updated_weights, original_loss


def train(epochs, inputs, outputs, weights, lr):
	"""
	模型训练
	:param epochs: 迭代次数
	:param inputs: 输入数据
	:param outputs: 输出结果（标准答案）
	:param weights: 权重参数
	:param lr: 学习率
	:return: 损失值列表，最后的权重参数
	"""
	losses = []
	# 执行 100 个 epoch 内执行前向传播和反向传播。
	for epoch in range(1, epochs + 1):
		weights, loss = update_weights(inputs, outputs, weights, lr)
		losses.append(loss)
	return losses, weights


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='NumPy 手写简单的神经网络')
	parser.add_argument('--lr', '-l', type=float, default=0.005, help='learning rate')
	parser.add_argument('--epochs', '-e', type=int, default=10000, help='number of epochs')
	parser.add_argument('--seed', '-s', type=int, default=2003, help='random seed')
	args = parser.parse_args()

	# 输入数据
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
	# 设置随机初始化种子
	np.random.seed(args.seed)
	# 随机初始化权重
	weights = [
		# 第一个参数数组对应于将输入层连接到隐藏层的权重矩阵
		np.random.random((2, 10)),
		# 第二个参数数组表示与隐藏层的每个神经元相关的偏置值
		np.random.random((1, 10)),
		# 第三个参数数组对应于将隐藏层连接到输出层 3x1 权重矩阵
		np.random.random((10, 1)),
		# 最后一个参数数组表示与输出层相关的偏执值。
		np.random.random(1)
	]

	# 模型训练
	losses, weights = train(args.epochs, inputs, outputs, weights, args.lr)

	# 获取更新后的权重后，通过将输入传递给网络对输入进行预测并计算输出值。
	test_data = np.array(list(product(range(1, 10), repeat=2)))
	# 标准输出结果
	test_ans = np.array([[x + y] for x, y in test_data])
	# 利用训练好的权重来预测测试数据
	pre_hidden = np.dot(np.array(test_data), weights[0]) + weights[1]
	hidden = 1 / (1 + np.exp(-pre_hidden))
	test_outputs = np.dot(hidden, weights[2]) + weights[3]
	# 结果四舍五入，不保留小数
	test_outputs_approximate = np.rint(test_outputs).astype(int)
	# 打印结果
	count = 0.0
	for input_data, approximate, output_data, stand_ans in zip(test_data, test_outputs_approximate, test_outputs, test_ans):
		input_data = list(map(int, input_data))  # 转成普通 int
		print(f"{input_data} -> {approximate[0]} ({output_data[0]})")
		count += 1 if approximate[0] == stand_ans[0] else 0
	print(f"模型预测的准确率为：{count / len(test_ans):.4f}")

	# 绘制损失值：
	plt.plot(losses)
	plt.title('Loss over increasing number of epochs')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()
