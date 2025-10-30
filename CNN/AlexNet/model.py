import torch
import torch.nn as nn


class AlexNet(nn.Module):
	# AlexNet类继承自 nn.Module，这是PyTorch中所有神经网络的基类。
	def __init__(self, num_classes=1000, init_weights=False):
		# 构造方法，'num_classes'表示分类的类别数，'init_wights'决定是否初始化权重
		super(AlexNet, self).__init__()
		"""
			特征提取层（卷积层和池化层的组合）。nn.Sequential是一个有序容器，按照定义的顺序执行每个层
			nn.Conv2d: 二维卷积层
				in_channels: 输入的通道数，RGB图像为3
				out_channels：输出的通道数
				kernel_size:卷积核的大小
				stride: 卷积核移动的步长
				padding: 填充，为了保持输入输出大小一致
			nn.RuLU: ReLU激活函数，表示直接在输入上进行操作，不额外占用内存
			nn.MaxPool2d: 最大池化层
				kernel_size: 池化窗口的大小
				stride: 池化窗口移动的步长
				padding： 填充
		"""
		self.feature = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),  # padding 能传入tuple或者int，表示不同地方补0
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
			nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=(2, 2)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
			nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=(1, 1)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
		)
		# 特征分类层（全连接层的组合）
		"""
			nn.Dropout(p=0.5): Dropout层，用于防止过拟合，p=0.5表示随机丢弃50%的神经元。
			nn.Linear: 全连接层。
				in_features=128*6*6: 输入的特征数，这里的128*6*6是经过前面卷积层和池化层后得到的特征图尺寸。
				out_features=2048: 输出的特征数。
			nn.ReLU(): ReLU激活函数。
		"""
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(in_features=128*6*6, out_features=2048),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(in_features=2048, out_features=2048),
			nn.ReLU(),
			nn.Linear(in_features=2048, out_features=num_classes)
		)
		if init_weights:
			self._initialize_weights()

	"""
		前向传播
			输入x首先经过特征提取层self.feature，然后扁平化为一维向量，最后经过分类层self.classifier得到输出。
	"""
	def forward(self, x):
		x = self.feature(x)
		x = torch.flatten(x, start_dim=1)  # 展平
		x = self.classifier(x)
		return x

	"""
		初始化权重
			对于卷积层nn.Conv2d，使用Kaiming正态初始化。
			对于全连接层nn.Linear，使用正态分布初始化，均值为0，标准差为0.01，同时将偏置设置为0。
	"""
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
