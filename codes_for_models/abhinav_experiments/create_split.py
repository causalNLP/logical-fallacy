import pandas as pd
from sklearn.model_selection import train_test_split
fallacy_all=pd.read_csv('../../data/edu_all.csv')[['source_article','updated_label']]
fallacy_train,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
fallacy_dev,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
fallacy_train.to_csv('../../data/edu_train.csv')
fallacy_dev.to_csv('../../data/edu_dev.csv')
fallacy_test.to_csv('../../data/edu_test.csv')