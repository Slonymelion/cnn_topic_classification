# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:31:40 2019

11-747 Assignment 1
@author: MomWithMe
"""
from collections import defaultdict
import time
import random
import torch
import datetime
import json
import torch.utils.data.dataloader as dataloader
import numpy as np


# dataset for dataloader
# Dataset class for generating data loader
class TopicDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, mode='train'):
        super(TopicDataset, self).__init__()
        # data is a list of tuples (sent, label)
        self.X = [x[0] for x in dataset]
        self.Y = [x[1] for x in dataset] if mode == 'train' else None
        self.num_of_samples = len(dataset)
        self.mode = mode

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx]) if self.mode == 'train' else self.X[idx]
    

# custom collate function for used in data loader
def TopicCollate(data):
    # data will be a list of tuples in the form of ([list of word indexes], tag)
    # sort the list by length of the sentence first
    data.sort(key=lambda x: len(x[0]))  # ascending order
    maxlen = len(data[-1][0])
    minlen = len(data[0][0])
    
    # pad shorter sentences with PAD symbol
    X = [w[0]+[PAD]*(maxlen-len(w[0])) for w in data]
    #X = [w[0][:minlen] for w in data]  # cut longer sentences
    X = torch.LongTensor(X)
    y = [x[1] for x in data]
    y = torch.LongTensor(y)
    
    return X, y
    
    
class CNNclass(torch.nn.Module):
    def __init__(self, params):
        super(CNNclass, self).__init__()

        nwords = params['nwords']
        emb_size = params['emb_size']
        num_filters = params['num_filters']
        window_sizes = params['window_sizes']
        ntags = params['ntags']
        pretrain = params['pretrain']
        wordvec = params['wordvec']
        dual_channel = params['dual_channel']
        
        self.dual_channel = dual_channel
        
        self.dropout = torch.nn.Dropout(params['dropout'])
        # use random word embedding or pretrained word embedding
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        if pretrain:
            self.embedding.weight = torch.nn.Parameter(read_wordvec(wordvec, nwords, emb_size))
        else: 
            # uniform initialization
            torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # dual channel feature
        if self.dual_channel:
            # embedding in static channel does not update
            self.embedding_static = torch.nn.Embedding(nwords, emb_size)
            self.embedding_static.weight = torch.nn.Parameter(self.embedding.weight.clone(),
                                                              requires_grad=False)
            
        # Network layers
        self.conv_layers = torch.nn.ModuleList()
        for win in window_sizes:
            conv = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=win, stride=1, padding=win, dilation=1, groups=1, bias=True)
            self.conv_layers.append(conv)
        self.relu = torch.nn.ReLU()
        in_size = len(self.conv_layers) * num_filters
        in_size = in_size * 2 if self.dual_channel else in_size
        self.projection_layer = torch.nn.Linear(in_features=in_size, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # nwords x emb_size
#        emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords
        emb = emb.permute(0, 2, 1)
        hall = []
        for conv in self.conv_layers:
            h = conv(emb)                       # 1 x num_filters x nwords
            # Do max pooling
            h = h.max(dim=2)[0]                         # 1 x num_filters
            h = self.relu(h)
            hall.append(h)                          # 3 x num_filters
        if self.dual_channel:
            emb2 = self.embedding_static(words)
#            emb2 = emb2.unsqueeze(0).permute(0, 2, 1)
            emb2 = emb2.permute(0, 2, 1)
            for conv in self.conv_layers:
                h = conv(emb2)
                h = h.max(dim=2)[0]
                h = self.relu(h)
                hall.append(h)
        hall = torch.cat(hall, dim=1)
#        print('hall shape:{}'.format(hall.shape))
        out = self.dropout(self.projection_layer(hall))              # size(out) = 1 x ntags
        return out


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
PAD = w2i["<pad>"]  # symbol for padding
UNK = w2i["<unk>"]  # symbol for unknown words


# function helpers for reading dataset and pretrained wordvec
def read_dataset(filename, mode='train', unk=0):
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            if mode == 'train':
                yield ([w2i[x] for x in words.split(" ")], t2i[tag])
            else:
                # treat unseen word as unk
                words = words.split(" ")
                w_idx = [unk] * len(words)
                for i in range(len(words)):
                    if words[i] in w2i:
                        w_idx[i] = w2i[words[i]]
                yield (w_idx, t2i[tag])


def read_wordvec(filename, nwords, nemb):
    embed_matrix = torch.rand((nwords, nemb)) * 0.5 - 0.25  # uniform between [-0.25, 0.25]
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')
            word, vec = word[0], [float(x) for x in word[1:]]
            if word in w2i:
                embed_matrix[w2i[word]] = torch.tensor(vec, dtype=torch.float)  # only update word vec for words in corpus
    return embed_matrix


# Read in the data
trainfile = 'topicclass_train.txt'
devfile = 'topicclass_valid.txt'
testfile = 'topicclass_test.txt'
test_result = 'results'
train = list(read_dataset(trainfile))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset(devfile, mode='dev', unk=UNK))


# Hyperparameters
EPOCHS = 20
params = {}
params['nwords'] = len(w2i)
params['num_filters'] = 100
params['window_sizes'] = [3, 4, 5]
params['ntags'] = len(t2i)
params['pretrain'] = True
params['dual_channel'] = False
params['wordvec'] = 'glove.6B.300d.txt'
params['emb_size'] = int(params['wordvec'].split('.')[-2][:-1])
params['dropout'] = 0.5
params['batch_size'] = 200
params['lr'] = 1e-4
anneal_patience = 2  # for learning rate annealing
early_stop_count = 3  # for early stopping
num_workers = 6
run_test = True

# create dataloader
trainclass = TopicDataset(train)
devclass = TopicDataset(dev)
trainloader = dataloader.DataLoader(trainclass, batch_size=params['batch_size'], shuffle=True, num_workers=num_workers, collate_fn=TopicCollate)
devloader = dataloader.DataLoader(devclass, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=TopicCollate)

# initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNclass(params).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])


# create log file and start training
datenow = datetime.datetime.now()
log_file = '-'.join([str(x) for x in [datenow.month, datenow.day, datenow.hour, datenow.minute]])
test_result = test_result + '-' + log_file + '.out'
log_file = 'model-'+ log_file + '.log'

with open(log_file, "w", encoding='utf-8') as log:
    # record the model hyperparameters
    log.write("Model parameters:\n")
    log.write(json.dumps(params))
    log.write("\n\n")
    
    # start training
    print('training start')
    old_val_loss = np.inf
    old_val_acc = 0
    patience = 0
    early_stop = 0
    for ITER in range(EPOCHS):
        # Perform training
        print('---- {}/{} EPOCHS ----'.format(ITER+1, EPOCHS))
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        start = time.time()

        for words, tag in trainloader:
            words_tensor = words.to(device)
            tag_tensor = tag.to(device)
            scores = model(words_tensor)
            predict = scores.argmax(dim=1)
            my_loss = criterion(scores, tag_tensor)
            train_loss += my_loss.item()
            train_correct += (predict==tag_tensor).sum().item()

            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

        msg = "iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
            ITER, train_loss / len(train), train_correct / len(train), time.time() - start)
        print(msg)
        log.write(msg+'\n')

        # Perform testing
        val_correct = 0.0
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for words, tag in devloader:
                words_tensor = words.to(device)
                tag_tensor = tag.to(device)
                scores = model(words_tensor)
                val_loss += criterion(scores, tag_tensor).item() * len(tag)
                predict = scores.argmax(dim=1)
                val_correct += (predict==tag_tensor).sum().item()

            msg = "iter %r: val loss=%.4f, val acc=%.4f" % (ITER, val_loss/len(dev), val_correct / len(dev))
            print(msg)
            log.write(msg+'\n')
            
            # learning rate annealing
            #val_loss /= len(dev)
            val_acc = val_correct/len(dev)
            #if val_loss >= old_val_loss:
            if val_acc <= old_val_acc:
                patience += 1
                print('validation acuracy decreased. patience +1.')
                if patience >= anneal_patience:
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2
                    optimizer.load_state_dict(optim_state)
                    print('no patience! decrease learning rate by 2 now.')
                    print('')
                    print('early stopping countdown... {}'.format(early_stop_count-early_stop))
                    early_stop += 1
                    if early_stop > early_stop_count:
                        print('too many annealig happened. training stopped.')
                        break
                    patience = 0
            else:
                #old_val_loss = val_loss
                old_val_acc = val_acc
 
                   
# generate labels on test dataset
if run_test:
    print('generating labels on test data')
    i2t = {v:u for (u, v) in t2i.items()}
    model.eval()
    with open(test_result, "w", encoding='utf-8') as outfile:
        with open(testfile, 'r', encoding='utf-8') as f:
            for line in f:
                _, line = line.lower().strip().split(" ||| ")
                words = line.split(" ")
                w_idx = [UNK] * len(words)
                for i in range(len(words)):
                    if words[i] in w2i:
                        w_idx[i] = w2i[words[i]]
                w_idx = torch.tensor([w_idx]).type(torch.LongTensor).to(device)
                scores = model(w_idx)
                predict = scores.argmax(dim=1).item()
                tag_pred = i2t[predict]
                # write to output
                outline = " ||| ".join([tag_pred, line])
                print(outline)
                outfile.write(outline)
    print("generation done")
                
