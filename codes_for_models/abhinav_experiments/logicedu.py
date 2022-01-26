import cv2
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AdamW,AutoModelForSequenceClassification
import pandas as pd
import random
from torch import nn
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import logging
import argparse

def get_unique_labels(df,label_col_name):
  candidate_labels=df[label_col_name].unique()
  labels_set=set()
  for label in candidate_labels:
    labels=label.split(';')
    if len(labels)>1:
      print(labels)
    labels = [x.strip() for x in labels]
    labels_set.update(labels)
  candidate_labels=list(labels_set)
  return candidate_labels

class MNLIDataset(Dataset):

  def __init__(self,tokenizer_path, train_df, val_df,label_col_name,map,test_df=None,fallacy=False,undersample_train=False,
               undersample_val=False,undersample_test=False,undersample_rate=0.04):
    self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    torch.manual_seed(0)
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.base_path = '/content/'
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    print("token ids",self.tokenizer.sep_token_id,self.tokenizer.cls_token_id)
    self.train_data = None
    self.val_data = None
    self.label_col_name=label_col_name
    self.map=map
    self.mappings=pd.read_csv("../../data/mappings.csv")
    if fallacy:
      self.unique_labels=get_unique_labels(pd.concat([train_df,val_df,test_df]),self.label_col_name)
      print(self.unique_labels)
    self.init_data(fallacy,undersample_train,undersample_val,undersample_test,undersample_ratio=undersample_rate)
    
  
  def convert_to_mnli(self,df,undersample=False,undersampling_rate=0.02):
    data=[]
    for i,row in df.iterrows():
      for label in self.unique_labels:
        entry= [row['source_article']]
        # print(label)
        if self.map=='base':
            entry.append("This is an example of %s logical fallacy" % label)
        elif self.map=='simplify':
            # print(label)
            simplified_label=list(self.mappings[self.mappings['Original Name']==label]['Understandable Name'])[0]
            # print("Simplified %s to %s" %(label,simplified_label))
            entry.append("This is an example of %s" % simplified_label)
        elif self.map=='description':
            description=list(self.mappings[self.mappings['Original Name']==label]['Description'])[0]
            entry.append("This is an example of %s" % description)
        elif self.map=='logical-form':
            form=list(self.mappings[self.mappings['Original Name']==label]['Logical Form'])[0]
            entry.append("This article matches the following logical form: %s" % form)
        if label==row[self.label_col_name]:
          entry.append("entailment")
        else:
          p=random.random()
          if undersample and p>undersampling_rate:
            continue
          entry.append("contradiction")
        data.append(entry)
      return pd.DataFrame(data, columns=['sentence1', 'sentence2', 'gold_label'])

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
      # pair_token_ids = premise_id + hypothesis_id
      print("max token id=",max(pair_token_ids))
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values
      if len(pair_token_ids)>MAX_LEN or len(segment_ids)>MAX_LEN or len(attention_mask_ids)>MAX_LEN:
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

def multi_acc(y_pred, y_test,flip=True):
  preds = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
  acc = (preds == y_test).sum().float() / float(y_test.size(0))
  if flip:
      y_pred=1-preds.cpu()
      y_test=1-y_test.cpu()
  else:
      y_pred = preds.cpu()
      y_test = y_test.cpu()
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

def train(model, dataset, optimizer,logger,save_path,epochs=5,ratio=0.04,positive_weight=12):
  train_loader,val_loader,_=dataset.get_data_loaders()
  min_val_loss=float('inf')
  loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([positive_weight,1,1]).float())
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
    flag=True
    if val_loss<min_val_loss:
        min_val_loss=val_loss
        logger.info("saving model")
        model.save_pretrained(save_path)
    else:
        flag=False
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_prec: {train_prec:.4f} train_rec: {train_rec:.4f}| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_prec: {val_prec:.4f} val_rec: {val_rec:.4f}')
    logger.info("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    if not flag:
        break

def eval1(model,test_loader,logger,flip=True):
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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger = get_logger()
    logger.info("device = %s",device)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help = "tokenizer path")
    parser.add_argument("-m", "--model", help = "model path")
    parser.add_argument("-w", "--weight", help = "Weight of entailment loss")
    parser.add_argument("-s", "--savepath", help = "Path to save logicedu model")
    parser.add_argument("-f","--finetune",help="Set this flag if you want to finetune the model on MNLI",default='F')
    parser.add_argument("-sf", "--savepath2", help="Path to save mnli model", default="mnli")
    parser.add_argument("-mp","--map",help="Map labels to this category")
    args = parser.parse_args()
    print(args)
    logger.info("initializing model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=3)
    model.to(device)

    if args.finetune=='T':
        logger.info("initializing mnli dataset")
        mnli_train = pd.read_csv("../../data/multinli_1.0/multinli_1.0_train.txt", sep='\t',
                                 skiprows=[24809, 33960, 75910, 100113, 150637, 158833, 173103, 178251, 221950, 286844,
                                           314109])[['gold_label', 'sentence1', 'sentence2']]
        mnli_train = mnli_train.dropna()
        mnli_dev_matched = pd.read_csv("../../data/multinli_1.0/multinli_1.0_dev_matched.txt", sep='\t')[['gold_label','sentence1','sentence2']]
        mnli_dev_mismatched = pd.read_csv("../../data/multinli_1.0/multinli_1.0_dev_mismatched.txt", sep='\t')[['gold_label','sentence1','sentence2']]
        mnli_dev = pd.concat([mnli_dev_matched, mnli_dev_mismatched], ignore_index=True, sort=False)
        mnli_dev = mnli_dev.dropna()
        mnli_dev = mnli_dev[mnli_dev.gold_label != '-']
        mnli_ds = MNLIDataset(args.tokenizer, mnli_train, mnli_dev, 'gold_label',fallacy=False)
        logger.info("finetune on mnli")
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        train(model, mnli_ds, optimizer, logger, args.savepath2, ratio=1, epochs=100, positive_weight=1)
        logger.info("reinit model")
        model=AutoModelForSequenceClassification.from_pretrained(args.savepath2)
        model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    logger.info("creating dataset")
    fallacy_all=pd.read_csv('../../data/edu_all.csv')[['source_article','updated_label']]
    fallacy_train,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
    fallacy_dev,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
    fallacy_ds=MNLIDataset(args.tokenizer,fallacy_train,fallacy_dev,'updated_label',args.map,fallacy_test,fallacy=True)
    print("checking length",len(fallacy_ds.tokenizer),model.config.vocab_size)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    logger.info("starting training")
    train(model, fallacy_ds, optimizer,logger,args.savepath,ratio=1,epochs=100,positive_weight=int(args.weight))

    model=AutoModelForSequenceClassification.from_pretrained(args.savepath,num_labels=3)
    model.to(device)
    logger.info("starting testing")
    _,_,test_loader=fallacy_ds.get_data_loaders()
    scores=eval1(model,test_loader,logger)
    logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f",scores[4],scores[5],scores[1],scores[2],scores[3])