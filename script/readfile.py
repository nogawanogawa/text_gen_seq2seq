# -*- coding: utf-8 -*-

import sudachipy
import pandas as pd
from lang import Lang


def readfile(filename):
    """ ファイルを読み込んで、含意関係を考慮してDataFrameを作る

    Args:
        filename ([type]): [description]

    Returns:
        [pandas.DataFrame]: [読み込んだcsvのDataFrame]
    """
    df = pd.read_csv(filename, index_col=0, header=None, sep=' ')

    # "×"の除外
    df = df[df[2] != "×"]

    # "上位→下位"のときに列の入れ替え
    df_1 = df[df[1].str.contains('上位→下位')]
    df_1 = df_1.iloc[:, [0,1,3,2]]

    # 対象を結合
    df_2 = df[~df[1].str.contains('上位→下位')]
    df = pd.concat([df_1, df_2]).iloc[:, 2:]

    return df

def readLangs(lang1, lang2, df, reverse=False):
    """ 言語モデルと文のペアの初期化

    Args:
        lang1 ([str]): [元になるコーパス名]
        lang2 ([str]): [生成ターゲットのコーパス名]
        df ([pandas.DataFrame]): [df]
        reverse (bool, optional): [description]. Defaults to False.

    Returns:
        [(Lang, Lang, List[List[str]])]: [(src, target, 文のペア)]
    """

    pairs = df.values.tolist()

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, df, reverse=False):
    """ dfからコーパスの辞書を作成する

    Args:
        lang1 ([type]): [description]
        lang2 ([type]): [description]
        df ([type]): [description]
        reverse (bool, optional): [description]. Defaults to False.

    Returns:
        [(Lang, Lang, List[List[str]])]: [(src, target, 文のペア)]
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, df, reverse=reverse)
    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

