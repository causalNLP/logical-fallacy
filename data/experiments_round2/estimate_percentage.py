import pandas as pd

fallacy_train = pd.read_csv('../../data/edu_train.csv')
fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
fallacy_test = pd.read_csv('../../data/edu_test.csv')
fallacy_all = pd.concat([fallacy_train, fallacy_dev, fallacy_test])
hits = 0
total = 0
for article in fallacy_all['masked_articles']:
    if 'MSK' in article:
        hits += 1
    total += 1
print(hits / total * 100, " percent")
