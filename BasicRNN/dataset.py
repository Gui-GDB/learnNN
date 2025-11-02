import torch
from torch.utils.data import Dataset, DataLoader

from BasicRNN.read_imdb import read_imdb_vocab, read_imdb_words
from constant import EMBEDDING_LENGTH, LETTER_MAP


class WordDataset(Dataset):
	def __init__(self, words, max_length, is_onehot=True):
		"""
		构造数据集
		:param words 单词数组
		:param max_length 表示单词的最大长度
		:param is_onehot 是不是 one-hot 编码
		"""
		super().__init__()
		self.words = words
		self.max_length = max_length
		self.is_onehot = is_onehot

	def __len__(self):
		"""
		获取数据集的长度。
		"""
		return len(self.words)

	def __getitem__(self, index):
		"""
		获取某项数据
		return the (one-hot) encoding vector of a word.
		"""
		word = self.words[index] + ' '
		if self.is_onehot:
			# [单词的最大长度, one-hot编码的长度]
			tensor = torch.zeros(self.max_length, EMBEDDING_LENGTH)
			for i in range(self.max_length):
				if i < len(word):
					tensor[i][LETTER_MAP[word[i]]] = 1
				else:
					# 短单词的填充部分应该全是空字符。
					tensor[i][0] = 1
		else:
			tensor = torch.zeros(self.max_length, dtype=torch.long)
			for i in range(len(word)):
				tensor[i] = LETTER_MAP[word[i]]
		return tensor


def get_dataloader_and_max_length(limit_length=None, is_onehot=True, is_vocab=True, batch_size=256):
	"""
	获取数据集的 DataLoader 以及数据集的最大长度
	:param limit_length 过滤掉过长的单词
	:param is_onehot 是否采用 One-hot 编码
	:param is_vocab 使用imdb.vocab词汇表
	:param batch_size 训练批次的大小
	"""
	if is_vocab:
		words = read_imdb_vocab()
	else:
		words = read_imdb_words(n_files=200)

	# 获取词汇表中单词的最大长度
	max_length = 0
	for word in words:
		max_length = max(max_length, len(word))

	# 过滤掉词汇表中过长的单词
	if limit_length is not None and max_length > limit_length:
		words = [w for w in words if len(w) <= limit_length]
		max_length = limit_length

	# for <EOS> (space)
	max_length += 1

	dataset = WordDataset(words, max_length, is_onehot)
	return DataLoader(dataset, batch_size=batch_size), max_length
