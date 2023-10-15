if __name__ == '__main__':
    #warnings.filterwarnings("ignore")
    #config = get_config()
   # train_model(config)


    #wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("vocab_size",vocab_size)