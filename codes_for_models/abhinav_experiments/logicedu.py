import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AdamW,AutoModelForSequenceClassification
import pandas as pd
import random
from torch import nn
import time
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import logging

def get_unique_labels(df,label_col_name):
  candidate_labels=df[label_col_name].unique()
  labels_set=set()
  for label in candidate_labels:
    labels=label.split(';')
    if(len(labels)>1):
      print(labels)
    labels = [x.strip() for x in labels]
    labels_set.update(labels)
  candidate_labels=list(labels_set)
  return candidate_labels

class MNLIDataset(Dataset):

  def __init__(self,tokenizer_path, train_df, val_df,label_col_name,test_df=None,fallacy=False,undersample_train=False,undersample_val=False,undersample_test=False,undersample_rate=0.04):
    self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    torch.manual_seed(0)
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.base_path = '/content/'
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    self.train_data = None
    self.val_data = None
    self.label_col_name=label_col_name
    if fallacy:
      self.unique_labels=get_unique_labels(pd.concat([train_df,val_df,test_df]),self.label_col_name)
      print(self.unique_labels)
    self.init_data(fallacy,undersample_train,undersample_val,undersample_test,undersample_ratio=undersample_rate)
    
  
  def convert_to_mnli(self,df,undersample=False,undersampling_rate=0.02):
    data=[]
    for i,row in df.iterrows():
      for label in self.unique_labels:
        entry=[]
        entry.append(row['source_article'])
        entry.append("This is an example of %s logical fallacy" %(label))
        if label==row[self.label_col_name]:
          entry.append("entailment")
        else:
          p=random.random()
          if undersample and p>undersampling_rate:
            continue
          entry.append("contradiction")
        data.append(entry)
    return pd.DataFrame(data, columns = ['sentence1', 'sentence2','gold_label'])
          
    

  def init_data(self,fallacy,undersample_train=False,undersample_val=False,undersample_test=False,undersample_ratio=0.02):
    if fallacy:
      self.train_df = self.convert_to_mnli(self.train_df,undersample=undersample_train,undersampling_rate=undersample_ratio)
      self.val_df = self.convert_to_mnli(self.val_df,undersample=undersample_val,undersampling_rate=undersample_ratio)
      self.test_df = self.convert_to_mnli(self.test_df,undersample=undersample_test,undersampling_rate=undersample_ratio)
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)
    self.test_data = self.load_data(self.test_df)

  def load_data(self, df):
    if df is None:
      return None
    
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['sentence1'].to_list()
    hypothesis_list = df['sentence2'].to_list()
    label_list = df['gold_label'].to_list()

    for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values
      if(len(pair_token_ids)>MAX_LEN or len(segment_ids)>MAX_LEN or len(attention_mask_ids)>MAX_LEN):
        continue
      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset
  
  def get_undersampled(self,ratio=0.04,batch_size=32, shuffle=True):
    train_df_entail=self.train_df[self.train_df.gold_label=='entailment']
    train_df_contradict=self.train_df[self.train_df.gold_label=='contradiction']
    train_df_contradict=train_df_contradict.sample(n=int(len(train_df_contradict)*ratio))
    df=pd.concat([train_df_contradict,train_df_entail])
    df=self.load_data(df)
    return DataLoader(
      df,
      shuffle=shuffle,
      batch_size=batch_size
    )
    
  def get_data_loaders(self, batch_size=32, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )
    if self.test_data:
      test_loader = DataLoader(
        self.test_data,
        shuffle=False,
        batch_size=len(self.unique_labels)
      )
    else:
      test_loader=None

    return train_loader, val_loader,test_loader

def get_logger(level='DEBUG'):
    logger = logging.getLogger()
    logger.handlers=[]
    file_log_handler = logging.FileHandler('logfile.log')
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    logger.setLevel(level)
    return logger

def multi_acc(y_pred, y_test):
  preds = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
  acc = (preds == y_test).sum().float() / float(y_test.size(0))
  y_pred=1-preds.cpu()
  y_test=1-y_test.cpu()
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for i in range(len(y_pred)): 
      if y_test[i]==y_pred[i]==1:
          TP += 1
      if y_pred[i]==1 and y_test[i]!=y_pred[i]:
          FP += 1
      if y_test[i]==y_pred[i]==0:
          TN += 1
      if y_pred[i]==0 and y_test[i]!=y_pred[i]:
          FN += 1
  if TP==0:
    return acc,0,0
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  return acc,precision,recall

import time

def get_metrics(logits,labels,threshold=0.5,sig=True,tensors=True):
  if sig:
    sig=nn.Sigmoid()
    preds=sig(logits)
    preds=(preds>threshold)
  else:
    preds=logits
  # print(preds.cpu().int(),labels.cpu().int())
  if tensors:
    y_true=labels.cpu().int()
    y_pred=preds.cpu().int()
  else:
    y_true=np.array(labels)
    y_pred=np.array(preds)
  temp = 0
  for i in range(y_true.shape[0]):
      temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
  accuracy=temp / y_true.shape[0]
  precision=sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples',zero_division=0)
  recall=sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples',zero_division=0)
  exact_match=sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
  micro_f1_score=sklearn.metrics.f1_score(y_true,y_pred,average='micro',zero_division=0)
  macro_f1_score=sklearn.metrics.f1_score(y_true,y_pred,average='macro',zero_division=0)
  return accuracy,precision,recall,exact_match,micro_f1_score,macro_f1_score

def train(model, dataset, optimizer,logger,epochs=5,ratio=0.04):  
  train_loader,val_loader,_=dataset.get_data_loaders()
  min_val_loss=float('inf')
  loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([12,1,1]).float())
  loss_fn.to(device)
  for epoch in range(epochs):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc=0
    total_train_prec=0
    total_train_rec=0
    if ratio<1:
      train_loader=dataset.get_undersampled(ratio)
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
      if batch_idx%10==0:
        logger.debug('%d %d',epoch,batch_idx)
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)

      _, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
      loss=loss_fn(prediction,labels)
      acc,prec,rec = multi_acc(prediction, labels)
      # print(acc,prec,rec)
      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.float()
      total_train_prec += prec
      total_train_rec += rec

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    train_prec = total_train_prec/len(train_loader)
    train_rec = total_train_rec/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    total_val_prec = 0
    total_val_rec = 0
    
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)
        
        _, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        loss=loss_fn(prediction,labels)
        acc,prec,rec = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.float()
        total_val_prec += prec
        total_val_rec += rec

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)
    val_rec = total_val_rec/len(val_loader)
    val_prec = total_val_prec/len(val_loader)
    if(val_loss<min_val_loss):
        min_val_loss=val_loss
        logger.info("saving model")
        model.save_pretrained("saved_models/electra_small_finetuned_logicedu")
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_prec: {train_prec:.4f} train_rec: {train_rec:.4f}| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_prec: {val_prec:.4f} val_rec: {val_rec:.4f}')
    logger.info("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def eval1(model,test_loader,logger):
  with torch.no_grad():
      all_preds=[]
      all_labels=[]
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(test_loader):
        logger.debug("%d",batch_idx)
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)
        if model=="random":
          prediction=torch.rand([15,3])
        else:
          _, prediction = model(pair_token_ids, 
                              token_type_ids=seg_ids, 
                              attention_mask=mask_ids, 
                              labels=labels).values()
        all_preds.append(torch.log_softmax(prediction, dim=1).argmax(dim=1))
        all_labels.append(labels)
      all_preds=1-torch.stack(all_preds)
      all_labels=1-torch.stack(all_labels)
      all_preds[all_preds<0]=0
      all_labels[all_labels<0]=0
      return get_metrics(all_preds,all_labels,sig=False)


#return accuracy,precision,recall,exact_match,micro_f1_score,macro_f1_score
if __name__ == "__main__":
    logger=get_logger()
    
    logger.info("creating dataset")
    fallacy_all=pd.read_csv('../../data/edu_all_updated.csv')[['source_article','updated_label']]
    fallacy_train,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
    fallacy_dev,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
    fallacy_ds=MNLIDataset("facebook/bart-large-mnli",fallacy_train,fallacy_dev,'updated_label',fallacy_test,fallacy=True)
    
    device="cuda"
    logger.info("initializing model")
    model=AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    # logger.info("starting training")
    # train(model, fallacy_ds, optimizer,ratio=1,epochs=1)

    logger.info("starting testing")
    _,_,test_loader=fallacy_ds.get_data_loaders()
    scores=eval1("random",test_loader,logger)
    logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f",scores[4],scores[5],scores[1],scores[2],scores[3])