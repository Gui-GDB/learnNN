import os
import re


def read_imdb(dir='../data/aclImdb', split='pos', is_train=True):
    """
    读取某一子目录下的所有影评文件
    :param dir 数据集存放的路劲
    :param split 区分是积极（pos）or消极（neg）的影评
    :param is_train 指定是训练集还是测试集
    return: 返回一个字符串列表（每条评论为一个元素）
    """
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    lines = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            lines.append(line)
    return lines


def read_imdb_words(dir='../data/aclImdb', split='pos', is_train=True, n_files=1000):

    """
    读取若干影评并把它们拼成一个大字符串，再做简单清洗并按空格拆成词（返回词列表）。
    :param dir 数据集存放的目录
    :param split 区分是积极（pos）or 消极（neg）的影评
    :param is_train 指定是训练集还是测试集
    :param n_files 指定需要读取的影评数量
    return: 返回一个词表
    """
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    all_str = ''
    for file in os.listdir(dir):
        if n_files <= 0:
            break
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            all_str += line
        n_files -= 1

    # todo 这里不用做一个去重的处理吗？
    # 过滤掉除 26 个字母和空格的这 27 个字符。
    words = re.sub(u'([^\u0020\u0061-\u007a])', '', all_str.lower()).split(' ')

    return words


def read_imdb_vocab(dir='../data/aclImdb'):
    """
    读取数据集自带的 imdb.vocab 文件，清洗并返回词表列表。
    :param dir 数据集存放的路径
    :return 返回词汇表中的所有词汇
    """
    fn = os.path.join(dir, 'imdb.vocab')
    with open(fn, 'rb') as f:
        word = f.read().decode('utf-8').replace('\n', ' ')
        # todo 这里不用做一个去重的处理吗？
        # 过滤掉除 26 个字母和空格的这 27 个字符。
        words = re.sub(u'([^\u0020\u0061-\u007a])', '', word.lower()).split(' ')
        filtered_words = [w for w in words if len(w) > 0]

    return filtered_words


def main():
    vocab = read_imdb_vocab()
    print(vocab[0])
    print(vocab[1])

    lines = read_imdb()
    print('Length of the file:', len(lines))
    print('lines[0]:', lines[0])
    words = read_imdb_words(n_files=100)
    print('Length of the words:', len(words))
    for i in range(5):
        print(words[i])


if __name__ == '__main__':
    main()
