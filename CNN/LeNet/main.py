import argparse
import os
import time

import torch
from torch import nn, optim

from model import LeNet
from utils import utils


def parse_args():
	# 创建 ArgumentParser 对象
	parser = argparse.ArgumentParser(description='LeNet 实现 MNIST 数据集的识别分类任务。')
	# 训练相关参数
	parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs ')
	parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	# 设备配置
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training ')
	# 随机种子和训练控制
	parser.add_argument('--seed', type=int, default=2003, help='Random seed for reproducibility')
	# 模型保存路径
	parser.add_argument('--model_save_dir', type=str, default='./models', help='dir to save the model')
	# 数据集名称
	parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name.')
	# 数据集存放路径
	parser.add_argument('--data_path', type=str, default='../../data', help='Dataset name.')
	# 解析命令行参数
	args = parser.parse_args()
	return args


def train_and_evaluate(model, train_loader, val_loader, args):
	# 如果目录不存在，则创建目录
	os.makedirs(args.model_save_dir, exist_ok=True)
	# 设置设备
	device = torch.device('cuda' if args.cuda else 'cpu')
	# 将模型移动到设备（GPU/CPU）
	model.to(device)

	# 定义损失函数, 交叉熵损失函数，用于分类任务
	criterion = nn.CrossEntropyLoss()
	# 定义优化器SGD
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# 模型在验证集上的最好结果
	best_val_acc = 0.0
	tic = time.time()
	# 训练过程
	for epoch in range(1, args.epochs + 1):
		# 设置模型为训练模式
		model.train()

		total_loss = 0  # 记录每个epoch的总损失
		correct = 0  # 记录训练集上的正确分类数量
		total = 0  # 记录训练集上的总样本数

		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)  # 将数据和标签移动到设备

			optimizer.zero_grad()  # 清除之前的梯度
			output = model(data)

			loss = criterion(output, target)  # 计算损失
			loss.backward()  # 反向传播，计算梯度
			optimizer.step()  # 更新模型参数

			total_loss += loss.item()  # 累加损失
			_, predicted = torch.max(output, 1)  # 预测类别（输出最大的索引）
			correct += (predicted == target).sum().item()  # 计算正确预测的数量
			total += target.size(0)  # 统计样本总数

		# 计算训练集上的平均损失和准确率
		avg_train_loss = total_loss / len(train_loader)
		train_accuracy = correct / total
		# 在验证集上评估模型
		val_accuracy = evaluate(model, val_loader, device)

		# 打印基础日志信息
		toc = time.time()
		interval = toc - tic
		minutes = int(interval // 60)
		seconds = int(interval % 60)
		print(f'Epoch [{epoch}/{args.epochs}], '
			  f'Time: {minutes:02d}:{seconds:02d}, '
			  f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
			  f'Validation Accuracy: {val_accuracy:.4f}')
		# 保存验证集上表现最好的模型
		if val_accuracy > best_val_acc:
			best_val_acc = val_accuracy
			model_name = str(args.dataset) + '_best_model.pth'
			torch.save(model.state_dict(), os.path.join(args.model_save_dir, model_name))
			print(f"Best model saved (Val Acc: {best_val_acc:.4f})")


def evaluate(model, val_loader, device):
	# 设置模型为评估模式
	model.eval()
	correct = 0
	total = 0
	# 不需要计算梯度
	with torch.no_grad():
		for data, target in val_loader:
			data, target = data.to(device), target.to(device)  # 将数据和标签移动到设备
			output = model(data)
			_, predicted = torch.max(output, 1)  # 预测类别（输出最大的索引）
			correct += (predicted == target).sum().item()  # 计算正确预测的数量
			total += target.size(0)  # 统计样本总数

	accuracy = correct / total  # 计算验证集上的准确率
	return accuracy


if __name__ == '__main__':
	# 解析命令行参数
	args = parse_args()

	# 设置cuda
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	print("using device: " + torch.cuda.get_device_name(0) if args.cuda else 'cpu')
	# 设置随机数种子，保证实验的可复现
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	# 获取数据加载器
	train_loader, val_loader = utils.get_data_MNIST(args.data_path, args.batch_size)
	# 初始化模型
	model = LeNet()

	# 训练并评估模型
	train_and_evaluate(model, train_loader, val_loader, args)
