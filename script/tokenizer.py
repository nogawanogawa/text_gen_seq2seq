# -*- coding: utf-8 -*-
from sudachipy import tokenizer, dictionary

class SudachiTokenizer():

    def __init__(self, mode="C"):
        if mode == "A":
            splitmode = tokenizer.Tokenizer.SplitMode.A
        if mode == "B":
            splitmode = tokenizer.Tokenizer.SplitMode.B
        if mode == "C":
            splitmode = tokenizer.Tokenizer.SplitMode.C
        
        self.mode = splitmode
    
    def get_token(self, source):
        """ 形態素解析(Suadchi)

        Args:
            source ([str]): [対象の文]

        Returns:
            [List[str]]: [形態素解析した単語のリスト]
        """
        tokenizer_obj = dictionary.Dictionary().create()
        result = [m.surface() for m in tokenizer_obj.tokenize(source, self.mode)]

        return result