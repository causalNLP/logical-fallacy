import jsonlines
import pandas as pd
import stanza
import bisect
import random

nlp = stanza.Pipeline(lang='en', processors='tokenize')
data_train = []
data_val = []
data_test = []
mapping = {
    "Fallacy of Relevance (red herring)": "fallacy of relevance",
    "Ad Hominem": "ad hominem",
    "Ad Populum": "ad populum",
    "Circular Reasoning": "circular reasoning",
    "Equivocation": "equivocation",
    "Fallacy of Credibility": "fallacy of credibility",
    "Fallacy of Emotion": "appeal to emotion",
    "Fallacy of Extension (straw man)": "fallacy of extension",
    "Fallacy of Relevance (red herring)": "fallacy of relevance",
    "False Causality": "false causality",
    "False Dilemma": "false dilemma",
    "Faulty Generalization": "faulty generalization",
    "Intentional": "intentional",
    "Logical Fallacy": "fallacy of logic"
}
paths = ['../../data/Jad_Liz_all.jsonl', '../../data/Safiyyah_Brighton_all.jsonl']
objects = []
for path in paths:
    with jsonlines.open(path) as reader:
        for obj in reader:
            objects.append(obj)

random.shuffle(objects)
# print(obj.keys())
for idx, obj in enumerate(objects):
    doc = nlp(obj['data'])
    # print(type(obj['data']))
    i = 0
    start_indices = []
    for sentence in doc.sentences:
        # print(sentence.text)
        while obj['data'][i] != sentence.text[0]:
            i += 1
        start_indices.append(i)
        i += 1
    for label in obj['label']:
        start = max(bisect.bisect_right(start_indices, label[0]) - 1, 0)
        end = min(bisect.bisect_left(start_indices, label[1]), len(doc.sentences) - 1)
        hot_area = doc.sentences[start:end]
        hot_area = [sentence.text for sentence in hot_area]
        ans = ' '.join(hot_area)
        if label[2] in mapping.keys():
            if idx < 100:
                data_train.append([obj['source_url'], ans, mapping[label[2]]])
            elif idx < 125:
                data_val.append([obj['source_url'], ans, mapping[label[2]]])
            else:
                data_test.append([obj['source_url'], ans, mapping[label[2]]])
climate_train = pd.DataFrame(data_train, columns=['original_url', 'source_article', 'logical_fallacies'])
climate_val = pd.DataFrame(data_val, columns=['original_url', 'source_article', 'logical_fallacies'])
climate_test = pd.DataFrame(data_test, columns=['original_url', 'source_article', 'logical_fallacies'])
climate_all = pd.concat([climate_train, climate_val, climate_test])
climate_train.to_csv('../../data/climate_train.csv')
climate_val.to_csv('../../data/climate_dev.csv')
climate_test.to_csv('../../data/climate_test.csv')
climate_all.to_csv('../../data/climate_all.csv')
