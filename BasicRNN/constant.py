# 定义了一组用于 字符编码（character-level encoding）的常量
# 定义了 字符和数字索引之间的映射关系

# 代表模型一共有多少个“可识别字符”：26个letter + space
EMBEDDING_LENGTH = 27
# 字符 ——> 数字的映射关系，将空格 ' ' 的编号设为 0
LETTER_MAP = {' ': 0}
# 反向映射表： 数字 ——> 字符：是给程序用来“查表”的（用于数字→字符反向映射）。
ENCODING_MAP = [' ']

for i in range(26):
    LETTER_MAP[chr(ord('a') + i)] = i + 1
    ENCODING_MAP.append(chr(ord('a') + i))
# 所有字母的列表：是给人或外部逻辑用来“看/遍历”的（表示有哪些字符）。
LETTER_LIST = list(LETTER_MAP.keys())

# 手动编写的几个测试用例
TEST_WORDS = [
    'apple', 'appll', 'appla', 'apply',
    'bear', 'beer', 'berr', 'beee',
    'car', 'cae', 'cat', 'cac', 'caq',
    'query', 'queee', 'queue', 'queen', 'quest', 'quess', 'quees'
]
