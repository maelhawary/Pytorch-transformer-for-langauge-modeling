import Dataset as dt
import torch
import torch.nn as nn
import config as config
from Transformer import Embedding_and_Positioning as encoding
from Transformer import Multi_attention_heads as Matt
from Transformer import Transformer_block as TRA
from Transformer import Transformer as Transformer
import os


# Apply the initialization to your model
def train(device,confi,dir):    #warnings.filterwarnings("ignore")
    #config = get_config()
   # train_model(config)    
    #introduce_device
    #wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    tokk=dt.Tokiniazation(chars)
    enc=tokk.tok_encode()
    # Train and test splits
    data = torch.tensor(enc(text), dtype=torch.long)
    n = int(confi['split']*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    ### git bach (B) from the dataset yith seq_len (T), these values can be changed in the configuration file
    train_input, train_tgt =dt.get_batch(train_data,confi['seq_len'],confi['batch_size'],device)#(B,T)
    val_input, val_tgt =dt.get_batch(val_data,confi['seq_len'],confi['batch_size'],device)
    model=Transformer.DecoderBlock(device,vocab_size)
    model=model.to(device)
    #print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=confi['lr'])
    for iter in range(confi['num_epochs']):
    
        pred, loss = model(train_input, train_tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()  
        # make a folder for saving the model 
        if not os.path.exists(dir):
            os.makedirs(dir)      
        if iter % 500 == 0:
            torch.save(model , dir+'model_iter_'+str(iter)+'.pth')
            torch.save(model.state_dict(), dir+'mode_state_iter_'+str(iter)+'.pt') 
            torch.save(optimizer.state_dict(), dir+'optimizer_state_dict'+str(iter)+'.pt')           
           
        # every once in a while evaluate the loss on train and val sets
        if iter % confi['eval_iter'] == 0 or iter == 2:
        #print(f"step {iter}: train loss {estimate_loss.loss(model,confi['eval_iter'],train_input, train_tgt):.4f},val loss {estimate_loss.loss(model,confi['eval_iter'],val_input, val_tgt):.4f}")
            print(f"step {iter}: train loss {loss:.4f} val loss {model(val_input, val_tgt)[1]:.4f}")
        
             


