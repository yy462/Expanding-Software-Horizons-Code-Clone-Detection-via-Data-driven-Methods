import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from createclone_bcb import createast, creategmndata, createseparategraph
import models
from torch_geometric.data import Data, DataLoader

# device=torch.device('cuda:0')
device = torch.device('cpu')
astdict, vocablen, vocabdict = createast()
treedict = createseparategraph(astdict, vocablen, vocabdict, device, mode='astandnext', nextsib=False, ifedge=False,
                               whileedge=False, foredge=False, blockedge=False, nexttoken=False, nextuse=False)
traindata, validdata, testdata = creategmndata('11', treedict, vocablen, vocabdict, device)
print(len(traindata))
# trainloder=DataLoader(traindata,batch_size=1)
num_layers = int(4)
batch_size = 32
model = models.GMNnet(vocablen, embedding_dim=100, num_layers=num_layers, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CosineEmbeddingLoss()
criterion2 = nn.MSELoss()


def create_batches(data):
    # random.shuffle(data)
    batches = [data[graph:graph + batch_size] for graph in range(0, len(data), batch_size)]
    return batches


def test(dataset):
    # model.eval()
    count = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        x1 = torch.tensor(x1, dtype=torch.long, device=device)
        x2 = torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1 != None:
            edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction = model(data)
        output = F.cosine_similarity(prediction[0], prediction[1])
        results.append(output.item())
        prediction = torch.sign(output).item()

        if prediction > 0 and label.item() == 1:
            tp += 1
            # print('tp')
        if prediction <= 0 and label.item() == -1:
            tn += 1
            # print('tn')
        if prediction > 0 and label.item() == -1:
            fp += 1
            # print('fp')
        if prediction <= 0 and label.item() == 1:
            fn += 1
            # print('fn')
    print(tp, tn, fp, fn)
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print('precision is none')
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print('recall is none')
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results


num_epochs = 10
epochs = trange(num_epochs, leave=True, desc="Epoch")
for epoch in epochs:  # without batching
    print(epoch)
    batches = create_batches(traindata)
    totalloss = 0.0
    main_index = 0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        optimizer.zero_grad()
        batchloss = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)
            # print(len(data))
            # for i in range(len(data)):
            # print(i)
            # data[i]=torch.tensor(data[i], dtype=torch.long, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 != None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            prediction = model(data)
            # batchloss=batchloss+criterion(prediction[0],prediction[1],label)
            cossim = F.cosine_similarity(prediction[0], prediction[1])
            batchloss = batchloss + criterion2(cossim, label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    # test(validdata)
    devresults = test(validdata)
    devfile = open('gmnbcbresult/' + 'astandnext' + '_dev_epoch_' + str(epoch + 1), mode='w')
    for res in devresults:
        devfile.write(str(res) + '\n')
    devfile.close()
    testresults = test(testdata)
    resfile = open('gmnbcbresult/' + 'astandnext' + '_epoch_' + str(epoch + 1), mode='w')
    for res in testresults:
        resfile.write(str(res) + '\n')
    resfile.close()
    # torch.save(model,'gmnmodels/gmnbcb'+str(epoch+1))
    # for start in range(0, len(traindata), args.batch_size):
    # batch = traindata[start:start+args.batch_size]
    # epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

'''for batch in trainloder:
    batch=batch.to(device)
    print(batch)
    quit()
    time_start=time.time()
    model.forward(batch)
    time_end=time.time()
    print(time_end-time_start)
    quit()'''
