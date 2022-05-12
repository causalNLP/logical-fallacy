import pandas as pd
from sklearn.model_selection import train_test_split

fallacy_all = pd.read_csv('~/PycharmProjects/kialoscraping/data/kialo_nodup.csv')[['text', 'types of fallacy']]
fallacy_train, fallacy_rem = train_test_split(fallacy_all, test_size=300, random_state=10)
fallacy_dev, fallacy_test = train_test_split(fallacy_rem, test_size=150, random_state=10)
fallacy_train.to_csv('~/PycharmProjects/kialoscraping/data/kialo_train1.csv')
fallacy_dev.to_csv('~/PycharmProjects/kialoscraping/data/kialo_dev1.csv')
fallacy_test.to_csv('~/PycharmProjects/kialoscraping/data/kialo_test1.csv')
