
import torch
import torch.nn as nn

def Embedding_and_Postional_endocing(x,vocab_size,d_model):
    self.embd=nn.Embedding(vocab_size, d_model)
    self.pos=nn.Embedding(vocab_size, d_model)
    


