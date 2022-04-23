import pandas as pd
import stanza
from logicedu import get_logger

nlp = stanza.Pipeline(lang='en', processors='tokenize', use_gpu=True)


def get_stats(df):
    sent_count = 0
    word_count = 0
    words = set()
    i = 0
    for text in df['source_article']:
        if isinstance(text, float):
            continue
        doc = nlp(text)
        logger.warn(i)
        i += 1
        sent_count += len(doc.sentences)
        for sent in doc.sentences:
            word_count += len(sent.words)
            for word in sent.words:
                words.add(word.text)
    return [len(df), sent_count, word_count, len(words)]


logger = get_logger(level='WARN')
train_df = pd.read_csv('../../data/climate_train_mh.csv')
dev_df = pd.read_csv('../../data/climate_dev_mh.csv')
test_df = pd.read_csv('../../data/climate_test_mh.csv')
all_df = pd.concat([train_df, dev_df, test_df])
results = [get_stats(train_df), get_stats(dev_df), get_stats(test_df), get_stats(all_df)]
print(results)
