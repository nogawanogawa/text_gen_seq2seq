# -*- coding: utf-8 -*-
from metaflow import FlowSpec, step

from readfile import readfile, prepareData
from encoder import EncoderRNN
from attnDecoderRNN import AttnDecoderRNN
from trainer import Trainer

INPUT = "INPUT"
OUTPUT = "OUTPUT"

class TextGenFlow(FlowSpec):

    @step
    def start(self):  
        print("Reading File...")
        text_df = readfile("data/entail_evaluation_set.txt")
        self.src, self.target, self.pairs = prepareData(INPUT, OUTPUT, text_df)
        self.next(self.init_network)

    @step
    def init_network(self):
        print("Initializing Network...")
        hidden_size = 256
        self.encoder = EncoderRNN(self.src.n_words, hidden_size)
        self.attn_decoder = AttnDecoderRNN(hidden_size, self.target.n_words, dropout_p=0.1)
        self.next(self.train)

    @step
    def train(self):
        print("Training...")
        self.trainer = Trainer(src=self.src, target=self.target, pairs=self.pairs)
        self.encoder, self.decoder = self.trainer.trainIters(encoder=self.encoder, decoder=self.attn_decoder, n_iters=75000)
        self.next(self.end)

    @step
    def end(self):
        print("Evaluation...")
        self.trainer.evaluateRandomly(encoder=self.encoder, decoder=self.attn_decoder)

if __name__ == '__main__':
    TextGenFlow()
