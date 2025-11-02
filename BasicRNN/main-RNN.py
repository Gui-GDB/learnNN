import argparse

import torch

from BasicRNN.train_test import train_rnn1, language_model_test, sample


def rnn(args):
    rnn1 = train_rnn1(args)
    # Or load the models
    # state_dict = torch.load(str(args.model_save_dir) + '/rnn1.pth', map_location=args.device)
    # rnn1 = RNN1().to(args.device)
    #
    # rnn1.load_state_dict(state_dict)

    rnn1.eval()
    language_model_test(rnn1)
    sample(rnn1)


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 设置设备
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("using device: " + torch.cuda.get_device_name(0) if args.cuda else 'cpu')
    # 设置随机数种子，保证实验的可复现
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    rnn(args)
    # rnn2(args)


def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='RNN 实现字母级语言模型')
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs ')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    # 设备配置
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training ')
    # 随机种子和训练控制
    parser.add_argument('--seed', type=int, default=2003, help='Random seed for reproducibility')
    # 模型保存路径
    parser.add_argument('--model_save_dir', type=str, default='./models', help='dir to save the model')
    # 一个单词的最大字符数
    parser.add_argument('--limit_length', type=int, default=19, help='max length of one word.')
    # 数据集存放路径
    parser.add_argument('--data_path', type=str, default='../data', help='Dataset name.')
    # 隐层的维度数
    parser.add_argument('--hidden_units', type=int, default=32)
    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
