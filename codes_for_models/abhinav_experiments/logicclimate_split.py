import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split

data = []
mapping={
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
    "False Dilemma" : "false dilemma",
    "Faulty Generalization": "faulty generalization",
    "Intentional": "intentional",
    "Logical Fallacy": "fallacy of logic"
}
paths = ['../../data/Jad_Liz_all.jsonl', '../../data/Safiyyah_Brighton_all.jsonl']
for path in paths:
    with jsonlines.open(path) as reader:
        for obj in reader:
            # print(obj.keys())
            for label in obj['label']:
                # print(label)
                # print(obj['data'][label[0]:label[1]])
                i = 0
                count = 0
                start = None
                end = None
                sentences = obj['data'].split('\n')
                for sentence in sentences:
                    if len(sentence) + count > label[0] and start is None:
                        start = i
                    if len(sentence) + count >= label[1]:
                        end = i
                        break
                    count += len(sentence) + 1
                    i += 1
                while start > 0:
                    start -= 1
                    if len(sentences[start]) != 0:
                        break
                while end < len(obj['data'].split('\n')) - 1:
                    end += 1
                    if len(sentences[end]) != 0:
                        break
                hot_area = obj['data'].split('\n')[start:end + 1]
                ans = '\n'.join(hot_area)
                if label[2] in mapping.keys():
                    data.append([obj['source_url'], ans, mapping[label[2]]])
climate_all = pd.DataFrame(data, columns=['original_url', 'source_article', 'logical_fallacies'])
climate_all.to_csv('../../data/climate_all.csv')
print(len(climate_all))
climate_train,climate_rem=train_test_split(climate_all,test_size=400,random_state=10)
climate_dev,climate_test=train_test_split(climate_rem,test_size=200,random_state=10)
climate_train.to_csv('../../data/climate_train.csv')
climate_dev.to_csv('../../data/climate_dev.csv')
climate_test.to_csv('../../data/climate_test.csv')