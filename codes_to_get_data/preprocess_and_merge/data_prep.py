import os
import sys

path = os.path.abspath(os.path.join(__file__, "../../../"))
# path = os.path.abspath(os.pardir)
sys.path.append(path)  # sys.path.append("../..")

from file_paths import FilePaths


class Constants(FilePaths):
    fallacy_type_alignment = {'Imprecise/Unclear': 'Inaccurate',
                              'Imprecise': 'Inaccurate',
                              'Unclear': 'Inaccurate',
                              'Confused': 'Inaccurate',

                              'Unbiased': 'Sound_reasoning',
                              'Insightful': 'Sound_reasoning',
                              'Accurate': 'Sound_reasoning',
                              'Neutral': 'Sound_reasoning',
                              'nan': 'Sound_reasoning',

                              'Exaggerating': 'Cherry-picking',
                              'Overstates_scientific_confidence': 'Cherry-picking',
                              'Clickbait_headline': 'Cherry-picking',

                              'Lack_of_context': 'Biased',
                              }


class DataReader:
    def __init__(self, data_type=['edu', 'climate'][-1]):
        self.data_type = data_type
        import pandas as pd

        self.raw_df = pd.read_csv(C.data_csv[data_type])
        # if data_type == 'edu': self.raw_df = self.raw_df.rename(columns={"A": "a", "B": "c"})

        self.key_sent = C.key_sent
        self.key_fallacy = C.key_fallacy
        self.test_size = C.test_size[data_type]
        self.dev_size = C.dev_size[data_type]

    def save_cleaned_file(self):
        df = self.cleaner(self.raw_df)
        print('[Info] Writing {} lines to {}'.format(len(df), C.article_csv_clean))
        df.to_csv(C.article_csv_clean, index=False)

    def gen_sentence_data(self):
        import pandas as pd

        from efficiency.function import set_seed
        set_seed()
        if self.data_type == 'climate':
            df = self.cleaner(self.raw_df, binarize=False)
        else:
            df = self.raw_df

        df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle in-place

        dev_dataset, test_dataset = [], []
        for split, dataset in [('dev', dev_dataset), ('test', test_dataset)]:
            path = C.save_path[self.data_type][split]
            dataset.append(pd.read_csv(path))
            print('[Info] Reading {} lines froom {}'.format(len(dataset), path))

        dev_dataset, test_dataset = dev_dataset[0], test_dataset[0]
        dev_or_test = pd.concat([dev_dataset, test_dataset])
        df = pd.concat([df, dev_or_test])
        train_dataset = df.drop_duplicates(keep=False).reset_index(drop=True)
        df = pd.concat([train_dataset, dev_or_test]).reset_index(drop=True)
        import pdb;
        pdb.set_trace()

        from efficiency.nlp import NLP
        nlp = NLP()

        stats = []
        for data in [df, train_dataset, dev_dataset, test_dataset]:
            articles = data[self.key_sent].tolist()
            articles = [[nlp.word_tokenize(sent).strip() for sent in nlp.sent_tokenize(art)] for art in articles]
            article_len = [sum(len(s.split()) for s in a) for a in articles]
            num_sents = [len(a) for a in articles]
            from efficiency.function import flatten_list
            vocab = flatten_list([' '.join(a).split() for a in articles])
            vocab = set(vocab)
            from efficiency.function import avg
            mean, std = avg(article_len, return_std=True)

            from efficiency.log import show_var
            show_var(['len(articles)', 'sum(num_sents)', 'sum(article_len)', 'mean', 'std', 'min(article_len)',
                      'max(article_len)'],
                     joiner=', ')
            show_var(['len(vocab)'])
            stats.append([len(articles), sum(num_sents), sum(article_len), len(vocab)])
        headers = ['\\textbf{Total Data}',
                   '\\quad Train',
                   '\\quad Dev',
                   '\\quad Test', ]
        printout = [' & '.join([h] + ['{:,}'.format(i) for i in row]) + ' \\\\' for h, row in zip(headers, stats)]
        printout = '\n'.join(printout)
        print(printout)

        import sys
        sys.exit()
        import pdb;
        pdb.set_trace()

    def dist_articles(self):
        from efficiency.function import set_seed
        set_seed()
        if self.data_type == 'climate':
            df = self.cleaner(self.raw_df, binarize=False)
        else:
            df = self.raw_df

        df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle in-place

        test_dataset = df.sample(n=self.test_size, random_state=0)
        train_n_dev_dataset = df.drop(test_dataset.index).reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        dev_dataset = train_n_dev_dataset.sample(n=self.dev_size, random_state=0)
        train_dataset = train_n_dev_dataset.drop(dev_dataset.index).reset_index(drop=True)
        dev_dataset = dev_dataset.reset_index(drop=True)

        for split, dataset in [('train', train_dataset), ('dev', dev_dataset), ('test', test_dataset)]:
            path = C.save_path[self.data_type][split]
            dataset.to_csv(path, index=False)
            print('[Info] Saving {} lines to {}'.format(len(dataset), path))
        import pdb;
        pdb.set_trace()

        from efficiency.nlp import NLP
        nlp = NLP()

        stats = []
        for data in [df, train_dataset, dev_dataset, test_dataset]:
            articles = data[self.key_sent].tolist()
            articles = [[nlp.word_tokenize(sent).strip() for sent in nlp.sent_tokenize(art)] for art in articles]
            article_len = [sum(len(s.split()) for s in a) for a in articles]
            num_sents = [len(a) for a in articles]
            from efficiency.function import flatten_list
            vocab = flatten_list([' '.join(a).split() for a in articles])
            vocab = set(vocab)
            from efficiency.function import avg
            mean, std = avg(article_len, return_std=True)

            from efficiency.log import show_var
            show_var(['len(articles)', 'sum(num_sents)', 'sum(article_len)', 'mean', 'std', 'min(article_len)',
                      'max(article_len)'],
                     joiner=', ')
            show_var(['len(vocab)'])
            stats.append([len(articles), sum(num_sents), sum(article_len), len(vocab)])
        headers = ['\\textbf{Total Data}',
                   '\\quad Train',
                   '\\quad Dev',
                   '\\quad Test', ]
        printout = [' & '.join([h] + ['{:,}'.format(i) for i in row]) + ' \\\\' for h, row in zip(headers, stats)]
        printout = '\n'.join(printout)
        print(printout)

        import sys
        sys.exit()
        import pdb;
        pdb.set_trace()

    def dist_logical_fallacies(self):
        df = self.cleaner(self.raw_df, binarize=False)
        fallacy_lists = df[self.key_fallacy].tolist()
        from efficiency.function import flatten_list
        num_samples = len(fallacy_lists)
        all_occurrences = flatten_list(fallacy_lists)
        from collections import Counter
        cnt = Counter(all_occurrences)
        cnt = [(k.replace('_', ' '), v / num_samples) for k, v in cnt.items()]
        cnt = sorted(cnt, key=lambda i: i[-1], reverse=True)
        printout = ['{} & {:.2f}\\% \\\\'.format(k.title(), 100 * v) for k, v in cnt]
        printout = '\n'.join(printout)
        print(printout)
        import pdb;
        pdb.set_trace()
        '''
    def get_statistics(self, data, key_label='logical_fallacies'):
        fallacy_lists = [i[key_label].split('; ') for i in data]

        from efficiency.function import flatten_list
        num_samples = len(fallacy_lists)
        all_occurrences = flatten_list(fallacy_lists)
        from collections import Counter
        cnt = Counter(all_occurrences)
        cnt = [(k.replace('_', ' '), v / num_samples) for k, v in cnt.most_common()]
        # cnt = sorted(cnt)

        printout = ['{} & {:.2f}\\% \\\\'.format(k.capitalize(), 100 * v) for k, v in cnt]
        printout = '\n'.join(printout)
        import pdb;
        pdb.set_trace()
        print(printout)
        show_var(['cnt'])
        
        '''

    def plot_logical_fallacies(self):
        df = self.cleaner(self.raw_df)
        fal_bin_df = df[df.columns[2:]]
        acov = fal_bin_df.T.dot(fal_bin_df)
        # import pdb;pdb.set_trace()
        # df[df['Misleading'] == 1]
        # df.loc[[58]]
        # df.loc[[69]]

        # get average # label per article
        # fal_bin_df.to_numpy().sum() / len(fal_bin_df)

        import numpy as np
        # acov[np.diag_indices_from(acov)] = 0
        np.fill_diagonal(acov.values, 0)
        acov /= acov.sum()
        # acov[np.diag_indices_from(acov)] = 1
        np.fill_diagonal(acov.values, 1)
        import pdb;
        pdb.set_trace()

        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        # im = ax.imshow(acov, vmin=0, vmax=1, cmap='OrRd')

        import seaborn as sns
        sns.heatmap(acov, cmap="OrRd")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.tight_layout()
        plt.show()

    def cleaner(self, df, binarize=True):
        df = df[['original_url', self.key_sent, self.key_fallacy]]
        df[self.key_fallacy] = df[self.key_fallacy].apply(lambda x: str(x).replace(' ', '_').split(',_'))
        df[self.key_sent] = df[self.key_sent].apply(lambda x: ' '.join(x.split()[:900]).rsplit('.', 1)[0] + '.')

        df[self.key_fallacy] = self.preprocess_logical_fallacies(df[self.key_fallacy])
        if binarize:
            import pandas as pd
            from sklearn.preprocessing import MultiLabelBinarizer

            mlb = MultiLabelBinarizer(sparse_output=True)
            df = df.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(df.pop(self.key_fallacy)),
                    index=df.index,
                    columns=mlb.classes_))

        df = df.dropna(subset=[self.key_sent])  # remove row with NaN in `source_article` column
        return df

    @staticmethod
    def preprocess_logical_fallacies(df_fallacies):
        for row_ix, fallacy_list in enumerate(df_fallacies):
            for list_ix, fallacy_type in enumerate(fallacy_list):
                if fallacy_type in C.fallacy_type_alignment:
                    try:
                        df_fallacies[row_ix][list_ix] = C.fallacy_type_alignment[fallacy_type]
                    except:
                        import pdb;
                        pdb.set_trace()
            df_fallacies[row_ix] = list(set(df_fallacies[row_ix]))

        return df_fallacies
        fallacies = df_fallacies.values
        from collections import Counter
        from efficiency.function import flatten_list
        all_occurrences = flatten_list(fallacies)
        cnt = Counter(all_occurrences)
        import pdb;
        pdb.set_trace()
        # Counter({'Soundreasoning': 78, 'Misleading': 67, 'Inaccurate': 43, 'Flawedreasoning': 39, 'Cherry-picking': 37, 'Biased': 32, 'Alarmist': 6, 'Inappropriatesources': 5, 'Derogatory': 5})


def main():
    dr = DataReader()
    # dr.gen_sentence_data()
    # dr.dist_articles()

    dr.dist_logical_fallacies()
    # dr.plot_logical_fallacies()


if __name__ == '__main__':
    C = Constants()
    main()
