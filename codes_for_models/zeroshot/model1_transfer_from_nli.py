class Constants:
    url_csv_sentence_logical_fallacies = "https://doc-14-14-sheets.googleusercontent.com/export/70cmver1f290kjsnpar5ku2h9g/i58lts2ik9nn8ft0mbaeqpivuc/1630142750000/106535178685861528249/*/1Mf3aaSmo4DUUsymgIIxm_Vl4Hd-wnGsO0WB3pbD_u-k?format=csv&amp;id=1Mf3aaSmo4DUUsymgIIxm_Vl4Hd-wnGsO0WB3pbD_u-k&amp;gid=0"
    data_folder = '../../data/'
    csv_sentence_logic = data_folder + 'sentence_logic.csv'
    csv_sentence_logic_with_neg_labels = data_folder + 'sentence_logic_with_neg_labels.csv'
    csv_article_logic = data_folder + 'onehot_9class_20210825.csv'
    label_connector = '; '

    csv_fallacy_n_explanation = data_folder + 'sentence_logic_fallacy_type_n_explanation.csv'
    num_labels_to_confuse = 10
    # sentence_logical_fallacy_types = ['Slippery Slope',
    #                                   'Hasty Generalization',
    #                                   'Post hoc ergo propter hoc',
    #                                   ]
    file_gpt3_api = data_folder + 'api_openai.txt'
    result_folder = 'zeroshot/results/'
    file_sent_0shot_gpt2_pred = result_folder + 'sent_0shot_gpt2.json'
    file_zeroshot_gpt3_pred = result_folder + 'result_0shot_gpt3.json'
    file_zeroshot_gen_gpt3_pred = result_folder + 'result_0shot_gen_gpt3.json'

    csv_article_feedback = 'data/ClimateFeedbackDataset_20210825.csv'


class DataReader:
    def __init__(self, data_type=['sentence', 'article'][0]):
        if data_type == 'sentence':
            self.read_sent_logic()
            import pdb;
            pdb.set_trace()
        elif data_type == 'article':
            self.read_article_logic()

        from efficiency.log import show_time
        show_time('Finished obtaining {} data'.format(data_type))

    def read_article_logic(self):
        C = Constants()

        path = C.csv_article_logic
        from efficiency.log import read_csv
        data = read_csv(path)
        header_sentence = 'source_article'
        header_labels = sorted([i for i in data[0].keys() if i not in ('source_article', 'original_url')])

        sentences = [i[header_sentence] for i in data]
        sentences = [i.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip() for i in sentences]
        labels = [[j.replace('_', ' ') for j in header_labels if i[j] == '1'] for i in data]
        neg_labels = [[j.replace('_', ' ') for j in header_labels if i[j] != '1'] for i in data]

        self.sentence_n_labels_n_neg_labels = list(zip(sentences, labels, neg_labels))

        self.label_set = set(header_labels)

    def read_sent_logic(self):
        # import pandas as pd
        # path = 'https://raw.githubusercontent.com/juliencohensolal/BankMarketing/master/rawData/bank-additional-full.csv'
        # path = C.url_csv_sentence_logical_fallacies
        # data = pd.read_csv(path)  # use sep="," for coma separation.
        # data.describe()

        import os
        from efficiency.log import read_csv

        C = Constants()

        file = C.csv_sentence_logic_with_neg_labels
        header = ['labels', 'negative_labels', 'sentence', ]

        if os.path.isfile(file):
            data = read_csv(file)
            self.sentence_n_labels_n_neg_labels = []
            all_labels = []
            for row in data:
                labels = row[header[0]].split(C.label_connector)
                neg_labels = row[header[1]].split(C.label_connector)
                sent = row[header[2]]

                self.sentence_n_labels_n_neg_labels.append((sent, labels, neg_labels))
                all_labels.extend(labels)
        else:
            from efficiency.function import set_seed
            set_seed(verbose=True)

            data = read_csv(C.csv_sentence_logic)
            header_sentence = [i for i in data[0].keys() if i.startswith('Sentence')][0]
            header_label = [i for i in data[0].keys() if i.startswith('Logical Fallacy Type')][0]

            sentences = [i[header_sentence] for i in data]
            sentences = [i.replace('\n', ' ').strip() for i in sentences]

            # Key separators: " & " to connect multiple logical fallacies, " < " to connect hierarchies of logical fallacy types, and " / " to connect synonyms of each other
            labels = [i[header_label] for i in data]

            from efficiency.function import flatten_list
            all_labels = flatten_list(labels)

            from efficiency.function import random_sample
            neg_labels = []
            for this_labels in labels:
                # show_var(['sent','labels'])
                this_neg_labels = sorted(random_sample(all_labels, size=C.num_labels_to_confuse + len(this_labels)))
                this_neg_labels = sorted(
                    [i for i in this_neg_labels if i not in set(this_labels)][:C.num_labels_to_confuse])
                neg_labels.append(this_neg_labels)

            self.sentence_n_labels_n_neg_labels = list(zip(sentences, labels, neg_labels))
            rows = [
                [C.label_connector.join(sorted(l)), C.label_connector.join(sorted(n)), s]
                for s, l, n in zip(sentences, labels, neg_labels)
            ]
            from efficiency.log import write_rows_to_csv
            write_rows_to_csv([header] + rows, C.csv_sentence_logic_with_neg_labels, verbose=True)

        from collections import Counter

        cnt = Counter(all_labels)
        from efficiency.log import show_var
        show_var(['cnt'])
        self.label_set_all = set(all_labels)

    def get_label_explanations(self):
        import os
        from efficiency.log import show_var
        C = Constants()

        file = C.csv_fallacy_n_explanation
        header = ['logical_fallacy_type', 'explanation']
        if os.path.isfile(file):
            from efficiency.log import read_csv
            label_n_explanation = read_csv(file, list_or_dict='list')[1:]
            for row_ix, row in enumerate(label_n_explanation):
                if len(row) != 2:
                    explanation = ','.join(row[1:]).replace('""', '"')
                    label_n_explanation[row_ix][1:] = [explanation]
            self.label2explanation = dict(label_n_explanation)
            return

        from efficiency.nlp import NLP
        import wikipedia
        from tqdm import tqdm

        nlp = NLP()

        label2explanation = {}
        bad_labels = []
        for label in tqdm(self.label_set_all):
            show_var(['label'])
            summary = ''
            page_name = wikipedia.search(label)[0]
            candidates = [label + ' fallacy', label, page_name]
            for candidate in candidates:
                try:
                    summary = wikipedia.summary(candidate)
                    break
                except:
                    pass

            if summary:
                explanation = nlp.sent_tokenize(summary)[0]
                show_var(['explanation', ])
                label2explanation[label] = explanation
            else:
                bad_labels.append(label)

        print(bad_labels)
        from efficiency.log import write_rows_to_csv
        writeout = sorted(label2explanation.items())
        write_rows_to_csv([header] + writeout, C.csv_fallacy_n_explanation, verbose=True)

        # ['red herring', 'nominal', 'plain folks and snob appeal', 'false continuum', 'disjunctive', 'golden mean', 'genetic', 'division', 'composition', 'straw man', 'non sequitur']
        import pdb;
        pdb.set_trace()

        self.label2explanation = label2explanation


class Classifier:
    def __init__(self, model_choice=['huggingface', 'flair', 'random'][0],
                 model=['base', 'facebook/bart-large-mnli', "roberta-large-mnli"][0]):
        self.model_choice = model_choice
        if model_choice == 'huggingface':
            from transformers import pipeline

            if model == 'base': model = 'facebook/bart-large-mnli'
            self.classifier = pipeline("zero-shot-classification", model=model, )
            # The pipeline can use any model trained on an NLI task, by default bart-large-mnli

        elif model_choice == 'flair':
            from flair.models.text_classification_model import TARSClassifier

            self.classifier = TARSClassifier.load('tars-base')
        from efficiency.log import show_time
        show_time("Finished initializing {}'s {} model".format(model_choice, model))

    def evaluate(self, data, label2explanation=None, data_type=['sentence', 'article'][0], multi_label=False):
        from tqdm import tqdm
        from collections import Counter
        from efficiency.log import show_var
        from efficiency.function import avg, random_sample, flatten_list
        hypothesis_template = "This " + data_type + " has the logical fallacy type {}"

        if label2explanation is not None:
            show_var(['len(label2explanation)'])

        _, all_labels, _ = zip(*data)
        all_labels = flatten_list(all_labels)
        majority = Counter(all_labels).most_common(1)[0][0]

        accuracies = []
        bar = tqdm(data)
        sent2preds = {}
        for sent, labels, neg_labels in bar:
            # show_var(['sent','labels'])
            all_labels = labels + neg_labels

            if label2explanation is not None:
                labels = ['{} ({})'.format(i, label2explanation.get(i, 'a type of logical fallacy'))
                          for i in labels]
                all_labels = ['{} ({})'.format(i, label2explanation.get(i, 'a type of logical fallacy'))
                              for i in all_labels]

            if self.model_choice == 'huggingface':
                results = self.classifier([sent], all_labels, hypothesis_template=hypothesis_template,
                                          multi_label=multi_label
                                          )
                predicted_label = {results[0]["labels"][0]}  # [0]

            elif self.model_choice == 'flair':
                # reference: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md
                from flair.data import Sentence

                sentence = Sentence(sent)
                self.classifier.predict_zero_shot(sentence, all_labels, multi_label=multi_label)
                sorted_labels = sorted(sentence.to_dict()['labels'], key=lambda k: k['confidence'], reverse=True)
                # predicted_label = sorted_labels[0]['value']
                predicted_label = {i['value'] for i in sorted_labels}

            elif self.model_choice == 'random':
                predicted_label = random_sample(all_labels, size=1)[0]
                predicted_label = {predicted_label}
            elif self.model_choice == 'majority':
                predicted_label = {majority}

            sent2preds[sent] = predicted_label
            acc = predicted_label <= set(labels)
            accuracies.append(acc)
            mean, std = avg(accuracies, return_std=True, decimal=4)

            import json
            # print(json.dumps(results, indent=4))
            # show_var(['acc', 'predicted_label', 'labels'])
            bar.set_description('{}, mean={:.2f}%'.format(self.model_choice, 100 * mean))
        show_var(['mean', 'std'])
        return sent2preds


class Tester:
    def run_all_models(self, data, data_type=['sentence', 'article'][0], expls=[None], multi_label=False,
                       model_ix=None):
        from efficiency.function import set_seed
        set_seed(verbose=True)

        from itertools import product
        model_choices = ['random', 'huggingface', 'flair', ]
        models = ['base', 'facebook/bart-large-mnli', "roberta-large-mnli"]
        model_choices_n_models = [
            ['majority', None],
            ['random', None],
            ['huggingface', 'roberta-large-mnli'],
            ['huggingface', 'facebook/bart-large-mnli'],
            ['flair', None],
        ][:-1]
        if model_ix is not None:
            model_choices_n_models = model_choices_n_models[model_ix: model_ix + 1]
        combinations = [[i] + m for i in expls for m in model_choices_n_models]

        from tqdm import tqdm
        setting2seq2preds = {}
        for label2explanation, model_choice, model in tqdm(combinations):
            print(model_choice, model)
            try:
                classifier = Classifier(model_choice=model_choice, model=model)
            except:
                # continue
                classifier = Classifier(model_choice=model_choice, model=model)

            preds = classifier.evaluate(data, label2explanation=label2explanation, data_type=data_type,
                                        multi_label=multi_label)
            setting = model.replace('facebook/', '') if model_choice == 'huggingface' else model_choice
            setting2seq2preds[setting] = preds
        return setting2seq2preds


def main():
    data_type = ['sentence', 'article'][0]
    dr = DataReader(data_type=data_type)
    dr.get_label_explanations()
    data = dr.sentence_n_labels_n_neg_labels

    from efficiency.log import show_var
    show_var(['C.num_labels_to_confuse'])
    print('[Info] Example data sample:', data[0])

    expls = [None, dr.label2explanation][:1]
    Tester().run_all_models(data, data_type, expls=expls)


if __name__ == '__main__':
    C = Constants()
    main()
