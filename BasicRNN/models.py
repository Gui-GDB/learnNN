import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import EMBEDDING_LENGTH, LETTER_LIST, LETTER_MAP


class RNN(nn.Module):
    def __init__(self, hidden_units=32):
        super().__init__()
        # hidden_units 表示隐藏层的神经元数目
        self.hidden_units = hidden_units
        # 这一行是将输入 x 和状态 a 拼接起来，所以这一层的输入通道数是 hidden_units + EMBEDDING_LENGTH。
        self.linear_a = nn.Linear(hidden_units + EMBEDDING_LENGTH, hidden_units)
        # 表示通过隐藏状态预测下一个字母的出现概率，因此这一层的输出通道是 EMBEDDING_LENGTH=27，即字符个数。
        self.linear_y = nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.tanh = nn.Tanh()

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]
        # 我们循环遍历的其实是单词长度那一维。为了方便理解代码，我们可以把单词长度那一维转置成第一维。
        # word shape: [max_word_length, batch,  embedding_length]
        word = torch.transpose(word, 0, 1)
        # 输出张量output[i][j]表示第j个batch的序列的第i个元素的27个字符预测结果。
        # output shape: [max_word_length, batch,  embedding_length]
        output = torch.empty_like(word)
        # 初始化好隐变量 a 和第一轮的输入 x。
        a = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        # 循环遍历序列的每一个字符，用 a x 计算 hat_y，并维护每一轮的 a x。
        # 最后，所有 hat_y 拼接成的 output 就是返回结果。
        for i in range(Tx):
            # 下一个隐变量和上一个隐变量 and 当前输入有关。
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            # 当前的输出只与当前的隐变量有关。
            hat_y = self.linear_y(next_a)
            output[i] = hat_y
            x = word[i]
            a = next_a

        # output shape: [batch, max_word_length, embedding_length]
        return torch.transpose(output, 0, 1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        """
        由于模型只能输出每一个单词的 softmax 前结果，所以为模型另外写一个求语言模型概率的函数。
        这个函数和 forward 大致相同。只不过，这次我们的输出 output 要表示每一个单词的概率。因此，它被初始化成一个全 1 的向量。
        """
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]
        # word shape: [max_word_length, batch,  embedding_length]
        word = torch.transpose(word, 0, 1)
        # word_label shape: [max_word_length, batch]
        word_label = torch.argmax(word, 2)

        # output shape: [batch]
        output = torch.ones(batch, device=word.device)

        a = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            tmp = self.linear_y(next_a)
            # 调用 softmax 得到概率值。
            hat_y = F.softmax(tmp, 1)
            probs = hat_y[torch.arange(batch), word_label[i]]
            output *= probs
            x = word[i]
            a = next_a

        return output

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        """
        采样单词
        根据语言模型输出的概率分布，采样出下一个单词；输入这一个单词，再采样下一个单词。这样一直采样，直到采样出空格为止。
        使用这种采样算法，我们能够让模型自动生成单词，甚至是英文里不存在，却看上去很像那么回事的单词。
        """
        batch = 1
        output = ''

        a = torch.zeros(batch, self.hidden_units, device=device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=device)
        for i in range(10):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            tmp = self.linear_y(next_a)
            hat_y = F.softmax(tmp, 1)

            np_prob = hat_y[0].detach().cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            x = torch.zeros(batch, EMBEDDING_LENGTH, device=device)
            x[0][LETTER_MAP[letter]] = 1
            a = next_a

        return output


class MyGRU(nn.Module):
    def __init__(self, hidden_size):
        super(MyGRU, self).__init__()
        self.input_size = EMBEDDING_LENGTH
        self.hidden_size = hidden_size

        # 拼接输入 [h_{t-1}, x_t] → 各个门
        self.linear_z = nn.Linear(hidden_size + EMBEDDING_LENGTH, hidden_size)
        self.linear_r = nn.Linear(hidden_size + EMBEDDING_LENGTH, hidden_size)
        self.linear_h = nn.Linear(hidden_size + EMBEDDING_LENGTH, hidden_size)

        # 表示通过隐藏状态预测下一个字母的出现概率，因此这一层的输出通道是 EMBEDDING_LENGTH=27，即字符个数。
        self.linear_y = nn.Linear(hidden_size, EMBEDDING_LENGTH)

    def forward(self, word):
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]
        # 我们循环遍历的其实是单词长度那一维。为了方便理解代码，我们可以把单词长度那一维转置成第一维。
        # word shape: [max_word_length, batch,  embedding_length]
        word = torch.transpose(word, 0, 1)
        # 输出张量output[i][j]表示第j个batch的序列的第i个元素的27个字符预测结果。
        # output shape: [max_word_length, batch,  embedding_length]
        output = torch.empty_like(word)
        # 初始化好隐变量 a 和第一轮的输入 x。
        h_t = torch.zeros(batch, self.hidden_size, device=word.device)
        x_t = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            combined = torch.cat([h_t, x_t], dim=1)  # 拼接
            z_t = torch.sigmoid(self.linear_z(combined))
            r_t = torch.sigmoid(self.linear_r(combined))

            # 对 r_t * h_{t-1} 与 x_t 再拼接
            combined_candidate = torch.cat([r_t * h_t, x_t], dim=1)
            h_tilde = torch.tanh(self.linear_h(combined_candidate))

            h_t = (1 - z_t) * h_t + z_t * h_tilde
            x_t = word[i]

            hat_y = self.linear_y(h_t)
            output[i] = hat_y

        # output shape: [batch, max_word_length, embedding_length]
        return torch.transpose(output, 0, 1)


class RNN2(torch.nn.Module):

    def __init__(self, hidden_units=64, embedding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(EMBEDDING_LENGTH, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_units, 1, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.hidden_units = hidden_units

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length]
        batch, Tx = word.shape[0:2]
        first_letter = word.new_zeros(batch, 1)
        x = torch.cat((first_letter, word[:, 0:-1]), 1)
        hidden = torch.zeros(1, batch, self.hidden_units, device=word.device)
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        y = self.decoder(output.reshape(batch * Tx, -1))

        return y.reshape(batch, Tx, -1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        batch, Tx = word.shape[0:2]
        hat_y = self.forward(word)
        hat_y = F.softmax(hat_y, 2)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            probs = hat_y[torch.arange(batch), i, word[:, i]]
            output *= probs

        return output

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        batch = 1
        output = ''

        hidden = torch.zeros(1, batch, self.hidden_units, device=device)
        x = torch.zeros(batch, 1, device=device, dtype=torch.long)
        for _ in range(10):
            emb = self.drop(self.encoder(x))
            rnn_output, hidden = self.rnn(emb, hidden)
            hat_y = self.decoder(rnn_output)
            hat_y = F.softmax(hat_y, 2)

            np_prob = hat_y[0, 0].detach().cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            x = torch.zeros(batch, 1, device=device, dtype=torch.long)
            x[0] = LETTER_MAP[letter]

        return output
