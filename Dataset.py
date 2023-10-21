# Here we #### load the dataset of a shakespeare note (tinyshakespeare) from (input.text), and then creating random batchs (B)
# for several sequences (T), so the input the decoder model is (B,T) corrsponding to (number of baches, sequence length)

import torch
import torch.nn as nn



class Tokiniazation():

    def __init__(self, chars: int) -> None:
        super().__init__
# create a mapping from characters to integers (i.e. Tokanization)
        self.chars=chars
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }      
    def tok_encode(self):
        encode_text = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
        return encode_text
    def tok_decode(self):
        decode_tok = lambda l: ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
        return decode_tok
    

# data loading
def get_batch(dt,T_len,B_len,device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(dt) - T_len, (B_len,))
    x = torch.stack([dt[i:i+T_len] for i in ix])
    y = torch.stack([dt[i+1:i+T_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y