# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:28:04 2020

author: AMS
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification, AlbertConfig
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from uuid import uuid4
from sklearn.metrics import confusion_matrix


config = AlbertConfig.from_pretrained('albert-base-v2')
config.num_labels = 6
print(config)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification(config)
df = pd.read_pickle('OPS/data/calidad.pkl')
dataPrueba1 = df[['OPS','score']]
dataPrueba1.loc[:,['score']]= (dataPrueba1['score']-1)/1.2

Xtrain,Xtest = train_test_split(dataPrueba1, train_size = 0.8, random_state = 42, stratify = dataPrueba1['score'] )
Xtrain = Xtrain.reset_index(drop = True)
Xtest = Xtest.reset_index(drop = True)

print(torch.cuda.is_available())



def prepare_features(seq_1, max_seq_length = 300, 
             zero_pad = False, include_CLS_token = True, include_SEP_token = True):

    tokens_a = tokenizer.tokenize(seq_1)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)

    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
 
    input_mask = [1] * len(input_ids)

    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
torch.backends.cudnn.benchmark=True
params = {'batch_size':1,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 0}

loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-05
optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)
class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        texto = self.data.OPS[index]
        label = self.data.score[index]
        X, _  = prepare_features(texto)
        return X, int(label)
    
    def __len__(self):
        return self.len
    
#model_path = "OPS/modelo_OPS/AlbertmodeloOPSBestV2.pth"
#model.load_state_dict(torch.load(model_path, map_location=device))

training_set = Intents(Xtrain)
testing_set = Intents(Xtest)


training_loader = DataLoader(training_set, **params)
testing_loader = DataLoader(testing_set, **params)


max_epochs = 1
model = model.train()
for epoch in tqdm(range(max_epochs)):
    print("EPOCH -- {}".format(epoch))
    for i, (sent, label) in enumerate(tqdm(training_loader)):
        optimizer.zero_grad()
        sent = sent.squeeze(0)
        if torch.cuda.is_available():
          sent = sent.cuda()
          label = label.cuda()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output, 1)
        
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            correct = 0
            total = 0
            Predlabels = []
            labels = []
            for sent, label in tqdm(testing_loader):
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                output = model.forward(sent)[0]
                labels.append(label)
                _, predicted = torch.max(output.data, 1)
                Predlabels.append(predicted)
                total += label.size(0)
                correct += (predicted.cpu() == label.cpu()).sum()
            accuracy = 100.00 * correct.numpy() / total
            print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
            
            
torch.save(model.state_dict(), 'Albert_state_dict_funcional'+ str(uuid4())+'.pth')

conf = confusion_matrix(labels,Predlabels)

def get_reply(msg):
  model.eval()
  input_msg, _ = prepare_features(msg)
  if torch.cuda.is_available():
    input_msg = input_msg.cuda()
  output = model(input_msg)[0]
  _, pred_label = torch.max(output.data, 1)
  prediction=(pred_label.data*1.2+1)
  prediction = prediction.tolist()
  return prediction


data = pd.read_excel("OPS/data/OPS MALT Ene-19 a Feb-20.xlsx")
dataOps = data["Observación positiva / oportunidad / brecha y cualquier medida inmediata adoptada"]

notas = []
for i in dataOps:
    notas.append(round(get_reply(i)[0],2))
    
salida = pd.DataFrame(columns= ['Creador',"OPS","Nota"])
salida['Creador'] = data['Creado por'].values
salida['OPS'] = dataOps.values
salida['Nota'] = notas

salida.to_excel('Notas.xlsx', index = False)
