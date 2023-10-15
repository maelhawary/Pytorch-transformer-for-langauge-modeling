# Here we #### load the dataset of a shakespeare note (tinyshakespeare) from (input.text), and then creating random batchs (B)
# for several sequences (T), so the input the decoder model is (B,T) corrsponding to (number of baches, sequence length)


class Tokiniazation():
    def __init__(self, chars) -> None:
        super().__init__
# create a mapping from characters to integers (i.e. Tokanization)
    self.stoi = { ch:i for i,ch in enumerate(chars) }
    self.itos = { i:ch for i,ch in enumerate(chars) }
    
    def tok_encode(self):
        encode_text = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        return encode_text
    def tok_decode(self):
        decode_tok = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
        return decode_tok
    
    
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y