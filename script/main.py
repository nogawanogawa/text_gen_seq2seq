# -*- coding: utf-8 -*-
import random

from readfile import readfile, prepareData
from encoder import EncoderRNN
from attnDecoderRNN import AttnDecoderRNN
from trainer import Trainer

INPUT = "INPUT"
OUTPUT = "OUTPUT"

# Read file 
print("Reading File...")
text_df = readfile("data/entail_evaluation_set.txt")
src, target, pairs = prepareData(INPUT, OUTPUT, text_df)
print(random.choice(pairs))

# Create network
print("Initializing Network...")
hidden_size = 256
encoder = EncoderRNN(src.n_words, hidden_size)
attn_decoder = AttnDecoderRNN(hidden_size, target.n_words, dropout_p=0.1)

# Train
print("Training...")
trainer = Trainer(src=src, target=target, pairs=pairs)
encoder, decoder = trainer.trainIters(encoder=encoder, decoder=attn_decoder, n_iters=75000)

# Evaluate
print("Evaluation...")
trainer.evaluateRandomly(encoder=encoder, decoder=attn_decoder)