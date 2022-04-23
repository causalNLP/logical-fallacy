import pandas as pd


def convert_to_multilabel(df, path):
    data = []
    for text in df['source_article'].unique():
        updated_table = df[df['source_article'] == text]['logical_fallacies']
        data.append([text, updated_table.tolist()])
    converted_df = pd.DataFrame(data, columns=['source_article', 'logical_fallacies'])
    converted_df.to_csv(path)


def get_ratio(df):
    pos_count = 0
    neg_count = 0
    for text in df['source_article'].unique():
        updated_table = df[df['source_article'] == text]['logical_fallacies']
        pos_count += len(updated_table)
        neg_count += 13 - len(updated_table)
    print(pos_count, neg_count, neg_count / pos_count)


fallacy_train = pd.read_csv('../../data/climate_train.csv')
fallacy_dev = pd.read_csv('../../data/climate_dev.csv')
fallacy_test = pd.read_csv('../../data/climate_test.csv')
fallacy_all = pd.concat([fallacy_train, fallacy_dev, fallacy_test])

dup_count = len(fallacy_all) - fallacy_all['source_article'].nunique()
# print(len(fallacy_all))
# print(fallacy_all['source_article'].nunique()
# print("%.2f" % dup_count / len(fallacy_all) * 100)

get_ratio(fallacy_train)
convert_to_multilabel(fallacy_train, '../../data/climate_train_mh.csv')
convert_to_multilabel(fallacy_test, '../../data/climate_test_mh.csv')
convert_to_multilabel(fallacy_dev, '../../data/climate_dev_mh.csv')
