import pandas as pd

train_df = pd.read_csv('../../data/climate_train.csv')
dev_df = pd.read_csv('../../data/climate_dev.csv')
test_df = pd.read_csv('../../data/climate_test.csv')
all_df = pd.concat([train_df, dev_df, test_df])
counts = all_df['logical_fallacies'].value_counts()
for count in counts.iteritems():
    print("%s & %.2f \\%% \\\\" % (count[0].title(), count[1]/len(all_df)*100))
