from logicedu import MNLIDataset
import gensim.downloader as api
import argparse
import pandas as pd
model = api.load('word2vec-google-news-300')
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tokenizer", help="tokenizer path")
parser.add_argument("-mp", "--map", help="Map labels to this category")
parser.add_argument("-ts", "--train_strat", help="Strategy number for training")
parser.add_argument("-ds", "--dev_strat", help="Strategy number for development and testing")
args = parser.parse_args()
fallacy_train = pd.read_csv('../../data/edu_train.csv')
fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
fallacy_test = pd.read_csv('../../data/edu_test.csv')
fallacy_ds = MNLIDataset(args.tokenizer, fallacy_train, fallacy_dev, 'new_label_2', args.map, fallacy_test,
                         fallacy=True, train_strat=int(args.train_strat), test_strat=int(args.dev_strat),
                         positive_class_weight=args.weight)
test_df = fallacy_ds.test_df
for i, row in df.iterrows():
    print(row['sentence1'], row['sentence2'], row['gold_label'])
    distance = model.wmdistance(row['sentence1'], row['sentence2'])
    print(distance)

