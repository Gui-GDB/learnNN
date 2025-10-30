from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, ImageFolder


def get_data_MNIST(data_path, batch_size):
	"""
	获取 MNIST 数据集，分为训练集和测试集使用
	:param data_path: 数据集存放的路径
	:param batch_size: 批处理的大小
	:return: 训练集和验证集加载器（DataLoader）
	"""
	# transforms.Compose()是一个容器，可以将多个转换操作组合在一起，返回一个组合后的转换操作。然后传递给 DataLoader。这样，当加载数据时，这些转换操作会自动应用到每一张图像上。
	transform = transforms.Compose([
		# 将PIL图像或NumPy数组转换为PyTorch张量，并自动将像素值归一化到 [0, 1]。
		transforms.ToTensor(),
		# 将每个像素值标准化，将其范围缩放到[-1.0, 1.0],有助于加速训练和提高网络的收敛速度
		transforms.Normalize((0.5,), (0.5,))
	])
	# 下载并加载训练集和测试集,train = True 加载训练集，否则加载测试集
	train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
	# 将这里的测试集，当作验证集使用
	val_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)
	# 创建数据加载器，用于加载数据集并将其分批次地传递给模型
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader


def get_data_dog_and_cat(data_path, batch_size, img_size):
	"""
	获取 dog_and_cat 数据集，分为训练集和测试集使用
	:param data_path: 数据集存放的路径
	:param batch_size: 批处理的大小
	:param img_size: 图片统一的尺寸大小
	:return: 训练集和验证集加载器（DataLoader）
	"""
	train_dir = data_path + '/dog_and_cat/training_set'
	test_dir = data_path + '/dog_and_cat/test_set'

	transform = transforms.Compose([
		# 统一图片尺寸
		transforms.Resize((img_size, img_size)),
		# 转为 [0,1]
		transforms.ToTensor(),
		# 将每个像素值标准化，将其范围缩放到[-1.0, 1.0],有助于加速训练和提高网络的收敛速度
		transforms.Normalize((0.5,), (0.5,))
	])

	train_dataset = ImageFolder(root=train_dir, transform=transform)
	test_dataset = ImageFolder(root=test_dir, transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader


if __name__ == '__main__':
	get_data_MNIST('../data', 256)
