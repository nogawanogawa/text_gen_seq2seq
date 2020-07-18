# -*- coding: utf-8 -*-
from tokenizer import SudachiTokenizer

SOS_token = 0
EOS_token = 1

class Lang:
    """単語管理のクラス

    INPUT/OUTPUTのコーパスに対して、単語の辞書を構築する

    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.sudachi = SudachiTokenizer()

    def addSentence(self, sentence):
        """辞書に文を登録する

        Args:
            sentence ([str]): [登録する文]
        """
        for word in self.sudachi.get_token(sentence):
            self.addWord(word)

    def addWord(self, word):
        """単語を登録する

        Args:
            word ([str]): [登録する単語]
        """
        if word not in self.word2index: # まだ登録されていない単語の場合

            # 単語の登録
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: # 登録済みの場合
            self.word2count[word] += 1 

