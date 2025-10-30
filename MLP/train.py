import os

import torch
import torch.optim as optim
import torch.nn as nn


def train_and_evaluate(model, train_loader, val_loader, args):
	# 如果目录不存在，则创建目录
	os.makedirs(args.model_save_dir, exist_ok=True)
	# 设置设备
	device = torch.device('cuda' if args.cuda else 'cpu')
	# 将模型移动到设备（GPU/CPU）
	model.to(device)

	# 定义损失函数, 交叉熵损失函数，用于分类任务
	criterion = nn.CrossEntropyLoss()
	# 定义优化器，Adam优化器
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	# 模型在验证集上的最好结果
	best_val_acc = 0.0

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
			output = model(data.view(data.size(0), -1))  # 扁平化输入数据，并进行前向传播

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

		print(f'Epoch [{epoch}/{args.epochs}], '
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
			output = model(data.view(data.size(0), -1))  # 扁平化输入数据，并进行前向传播
			_, predicted = torch.max(output, 1)  # 预测类别（输出最大的索引）
			correct += (predicted == target).sum().item()  # 计算正确预测的数量
			total += target.size(0)  # 统计样本总数

	accuracy = correct / total  # 计算验证集上的准确率
	return accuracy
