1. Download [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
    - This dataset was originally intended for the relatively simple NLP task of sentiment classification. It is definitely fine to use it to build an alphabetic language model.

2. Modify the directory in `read_imdb`.

3. Run `main.py` to train and test the language model. You can:

- Use `rnn1` or `rnn2`
- Switch the dataset by modifying `is_vocab` parameter of `get_dataloader_and_max_length`
- Tune the hyperparameters

to do more experiments.

- reference
   - [你的第一个PyTorch RNN模型——字母级语言模型](https://zhouyifan.net/2022/09/21/DLS-note-14-2/)
