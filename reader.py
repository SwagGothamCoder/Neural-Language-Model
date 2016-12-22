import numpy as np

def Read(path):
    text = ''
    print('Reading text...')
    with open(path, 'rt') as f:
        text = f.read()
    print('Done downloading.')
    text = text.split('\n')
    embed = {}
    vocab = []
    ivocab = {}
    ivocab['<PAD>'] = 0
    print('Organizing into embed and vocab structures...')
    for line in text:
        if not line.split(' ')[0]:
            print('Error')
        else:
            next_vec = [float(val) for val in line.split(' ')[1:]]
            embed[line.split(' ')[0]] = next_vec
            vocab += [line.split(' ')[0]]
    print('Creating ivocab...')
    for idx, word in enumerate(vocab):
        ivocab[word] = idx+1
    return vocab, ivocab, embed