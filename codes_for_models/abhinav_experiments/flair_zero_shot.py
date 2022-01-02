from flair.models import TARSClassifier
from flair.data import Sentence
import pandas as pd
from sklearn.model_selection import train_test_split
from logicedu import get_logger,get_unique_labels,get_metrics
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

if __name__ == "__main__":
    tars = TARSClassifier.load('tars-base')
    logger=get_logger()
    fallacy_all=pd.read_csv('../../data/edu_all_updated.csv')[['source_article','updated_label']]
    _,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
    _,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
    classes=get_unique_labels(fallacy_all,'updated_label')
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    print(mlb.classes_)
    labels=[]
    preds=[]
    for index,row in tqdm(fallacy_test.iterrows()):
        sentence = Sentence(row['source_article'])
        tars.predict_zero_shot(sentence, classes)
        sorted_labels = sorted(sentence.to_dict()['labels'], key=lambda k: k['confidence'], reverse=True)
        # predicted_label = sorted_labels[0]['value']
        predicted_label = {i['value'] for i in sorted_labels}
        labels_mh=mlb.transform([[row['updated_label']]])
        preds_mh=mlb.transform([list(predicted_label)])
        labels.append(labels_mh[0])
        preds.append(preds_mh[0])
    scores=get_metrics(preds,labels,sig=False,tensors=False)
    logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f",scores[4],scores[5],scores[1],scores[2],scores[3])

