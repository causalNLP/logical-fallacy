import torch
from transformers import AutoTokenizer, AdamW,AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
import pandas as pd
import argparse
from logicedu import get_logger,MNLIDataset

def eval1(model,test_loader,logger,unique_labels):
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
      results=[]
      for i in range(all_preds.shape[1]):
          preds=all_preds[:,i].cpu()
          labels=all_labels[:,i].cpu()
          prec=precision_score(labels,preds)
          rec=recall_score(labels,preds)
          f1=f1_score(labels,preds)
          no_of_p_labels=int(torch.sum(labels))
          results.append([unique_labels[i],prec,rec,f1,no_of_p_labels])
      return results

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device = ", device)
logger = get_logger()
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tokenizer", help="tokenizer path")
parser.add_argument("-m", "--model", help="model path")
args = parser.parse_args()
print(args)
logger.info("initializing model")
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
model.to(device)
fallacy_all=pd.read_csv('../../data/edu_all_updated.csv')[['source_article','updated_label']]
fallacy_train,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
fallacy_dev,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
fallacy_ds=MNLIDataset(args.tokenizer,fallacy_train,fallacy_dev,'updated_label',fallacy_test,fallacy=True)
logger.info("starting testing")
_, _, test_loader = fallacy_ds.get_data_loaders()
scores = eval1(model, test_loader,logger,fallacy_ds.unique_labels)
df = pd.DataFrame.from_records(scores,columns=['Fallacy Name','Precision','Recall','F1','Number of Positive Labels for this Class in Test Set'])
print(scores)
df.to_csv('classwise_electra.csv')