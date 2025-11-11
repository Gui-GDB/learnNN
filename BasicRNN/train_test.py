import os
from typing import Sequence, Tuple

import torch

from BasicRNN.constant import LETTER_MAP, EMBEDDING_LENGTH, TEST_WORDS
from BasicRNN.dataset import get_dataloader_and_max_length
from BasicRNN.models import RNN, MyGRU


def train_rnn1(args):
	# 获取数据集和相关信息
	dataloader, max_length = get_dataloader_and_max_length(args.limit_length, args.batch_size)
	# 初始化模型并且将模型移动到设备（CPU/GPU）
	model = RNN(args.hidden_units).to(args.device)
	# 定义优化器，Adam优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	# # 定义损失函数, 交叉熵损失函数，用于分类任务
	iteration = torch.nn.CrossEntropyLoss()

	for epoch in range(args.epochs):
		model.train()  # 设置模型为训练模式
		loss_sum = 0  # 记录每个epoch的总损失

		for y in dataloader:
			y = y.to(args.device)  # 将输入数据移动到指定设备上
			hat_y = model(y)  # [batch数, 最大单词长度, 字符数]
			n, Tx, _ = hat_y.shape

			# 交叉熵损失函数模型 hat_y 的维度是 [batch数， 类型数]，label_y 是一个一维整形标签数组。
			# 将 hat_y 的前两个维度融合在一起。
			hat_y = torch.reshape(hat_y, (n * Tx, -1))
			y = torch.reshape(y, (n * Tx, -1))

			# 准备 label_y 标签数组， 调用 argmax 把 one-hot 编码转换回标签。
			label_y = torch.argmax(y, 1)
			loss = iteration(hat_y, label_y)

			# 调用 PyTorch 的自动求导功能。
			optimizer.zero_grad()
			loss.backward()
			# 为了防止 RNN 梯度过大，使用 clip_grad_norm 截取梯度的最大值。
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			loss_sum += loss

		print(f'Epoch {epoch}. loss: {loss_sum / len(dataloader.dataset)}')

	# 如果目录不存在，则创建目录
	os.makedirs(args.model_save_dir, exist_ok=True)
	torch.save(model.state_dict(), str(args.model_save_dir) + '/rnn1.pth')
	return model


def train_gru(args):
	# 获取数据集和相关信息
	dataloader, max_length = get_dataloader_and_max_length(args.limit_length, args.batch_size)
	# 初始化模型并且将模型移动到设备（CPU/GPU）
	model = MyGRU(args.hidden_units).to(args.device)
	# 定义优化器，Adam优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	# # 定义损失函数, 交叉熵损失函数，用于分类任务
	iteration = torch.nn.CrossEntropyLoss()

	for epoch in range(args.epochs):
		model.train()  # 设置模型为训练模式
		loss_sum = 0  # 记录每个epoch的总损失

		for y in dataloader:
			y = y.to(args.device)  # 将输入数据移动到指定设备上
			hat_y = model(y)  # [batch数, 最大单词长度, 字符数]
			n, Tx, _ = hat_y.shape

			# 交叉熵损失函数模型 hat_y 的维度是 [batch数， 类型数]，label_y 是一个一维整形标签数组。
			# 将 hat_y 的前两个维度融合在一起。
			hat_y = torch.reshape(hat_y, (n * Tx, -1))
			y = torch.reshape(y, (n * Tx, -1))

			# 准备 label_y 标签数组， 调用 argmax 把 one-hot 编码转换回标签。
			label_y = torch.argmax(y, 1)
			loss = iteration(hat_y, label_y)

			# 调用 PyTorch 的自动求导功能。
			optimizer.zero_grad()
			loss.backward()
			# 为了防止 RNN 梯度过大，使用 clip_grad_norm 截取梯度的最大值。
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			loss_sum += loss

		print(f'Epoch {epoch}. loss: {loss_sum / len(dataloader.dataset)}')

	# 如果目录不存在，则创建目录
	os.makedirs(args.model_save_dir, exist_ok=True)
	torch.save(model.state_dict(), str(args.model_save_dir) + '/gru.pth')
	return model


def words_to_label_array(words: Tuple[str, Sequence[str]], max_length):
	if isinstance(words, str):
		words = [words]
	words = [word + ' ' for word in words]
	batch = len(words)
	tensor = torch.zeros(batch, max_length, dtype=torch.long)
	for i in range(batch):
		for j, letter in enumerate(words[i]):
			tensor[i][j] = LETTER_MAP[letter]

	return tensor


def words_to_onehot(words: Tuple[str, Sequence[str]], max_length):
	if isinstance(words, str):
		words = [words]
	words = [word + ' ' for word in words]
	batch = len(words)
	tensor = torch.zeros(batch, max_length, EMBEDDING_LENGTH)
	for i in range(batch):
		word_length = len(words[i])
		for j in range(max_length):
			if j < word_length:
				tensor[i][j][LETTER_MAP[words[i][j]]] = 1
			else:
				tensor[i][j][0] = 1

	return tensor


def onehot_to_word(arr):
	length, emb_len = arr.shape
	out = []
	for i in range(length):
		for j in range(emb_len):
			if arr[i][j] == 1:
				out.append(j)
				break
	return out


def language_model_test(model, is_onehot=True, device='cuda:0', limit_length=19):
	_, max_length = get_dataloader_and_max_length(limit_length)
	if is_onehot:
		test_word = words_to_onehot(TEST_WORDS, max_length)
	else:
		test_word = words_to_label_array(TEST_WORDS, max_length)
	test_word = test_word.to(device)
	probs = model.language_model(test_word)
	for word, prob in zip(TEST_WORDS, probs):
		print(f'{word}: {prob}')


def sample(model):
	words = []
	for _ in range(20):
		word = model.sample_word()
		words.append(word)
	print(*words)
