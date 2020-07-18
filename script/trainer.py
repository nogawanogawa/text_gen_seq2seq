# -*- coding: utf-8 -*-
import time
import math
import random
import torch
import torch.nn as nn
from torch import optim
from tokenizer import SudachiTokenizer

SOS_token = 0
EOS_token = 1

INPUT = "INPUT"
OUTPUT = "OUTPUT"
MAX_LENGTH = 30

class Trainer:
    def __init__(self, src, target, pairs):
        self.teacher_forcing_ratio = 0.5
        self.src = src
        self.target = target
        self.pairs = pairs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def asMinutes(self, s):
        """ 秒 -> 分に変換

        Args:
            s ([int]): [秒]

        Returns:
            [int]: [分]
        """
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        """ 時間の測定

        Args:
            since ([float]): [開始時刻]
            percent ([float]): [全体のうちいまどれだけ計算しているかの割合]

        Returns:
            [str]: [現在の所要時間 (- 予定終了所要時間) ]
        """
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def indexesFromSentence(self, lang, sentence):
        """ 文から単語のインデックスを取得する

        Args:
            lang ([Lang]): [コーパスをもとにした単語辞書]
            sentence ([str]): [対象の文]

        Returns:
            [type]: [description]
        """
        sudachi = SudachiTokenizer()
        return [lang.word2index[word] for word in sudachi.get_token(sentence)]


    def tensorFromSentence(self, lang, sentence):
        """ 文からEmbeddingを作る

        Args:
            lang ([type]): [description]
            sentence ([type]): [description]

        Returns:
            [tensor]: [単語のインデックス配列をEmbeddingの系列(1xn)として使用]
        """
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)


    def tensorsFromPair(self, pair):
        """ src / targetからテンソルを取得

        Args:
            pair ([List[str]]): [文のペア]

        Returns:
            [(tensor, tensor)]: [(srcのテンソル, targetのテンソル)]
        """
        input_tensor = self.tensorFromSentence(self.src, pair[0])
        target_tensor = self.tensorFromSentence(self.target, pair[1])
        return (input_tensor, target_tensor)


    def train(self, input_tensor, target_tensor, 
                encoder, decoder, encoder_optimizer, 
                decoder_optimizer, criterion, max_length=MAX_LENGTH):
        """ 1 iterarion分だけ学習する

        Args:
            input_tensor ([type]): [description]
            target_tensor ([type]): [description]
            encoder ([type]): [description]
            decoder ([type]): [description]
            encoder_optimizer ([type]): [description]
            decoder_optimizer ([type]): [description]
            criterion ([type]): [description]
            max_length ([type], optional): [description]. Defaults to MAX_LENGTH.

        Returns:
            [float]: [loss]
        """

        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length
    
    def trainIters(self, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        """ 学習する

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
            n_iters ([type]): [description]
            print_every (int, optional): [description]. Defaults to 1000.
            plot_every (int, optional): [description]. Defaults to 100.
            learning_rate (float, optional): [description]. Defaults to 0.01.

        Returns:
            [(Encoder, Decoder)]: [Encoder, Decoder]
        """
        encoder.to(self.device)
        decoder.to(self.device)
        
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [self.tensorsFromPair(random.choice(self.pairs))
                        for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            print(iter)
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
        
        return encoder, decoder


    def evaluate(self, encoder, decoder, sentence, max_length=MAX_LENGTH):
        """ モデルを評価する

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
            sentence ([type]): [description]
            max_length ([type], optional): [description]. Defaults to MAX_LENGTH.

        Returns:
            [(str, tensor)]: [(モデルによって生成された文, attension)]
        """
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(self.src, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                        encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.target.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]
    
    def evaluateRandomly(self, encoder, decoder, n=10):
        """ ペアからランダムにサンプリングして評価する

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
            n (int, optional): [description]. Defaults to 10.
        """

        encoder.to(self.device)
        decoder.to(self.device)

        for i in range(n):
            pair = random.choice(self.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
            
            