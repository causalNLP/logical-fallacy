class FilePaths:
    preprocess_data_folder = 'codes_to_get_data/intermediate_data_files/'
    raw_data_quiz_file_pattern = preprocess_data_folder + '20210901_*data*.csv'
    raw_data_quiz = preprocess_data_folder + 'sentence_logic_quiz.csv'
    raw_data_manual = preprocess_data_folder + 'sentence_logic_manual.csv'
    raw_data_logicallyfallacious_csv = preprocess_data_folder + 'sentence_logic_from_logicalfallacious.csv'
    raw_data_propaganda = 'propaganda_dataset/train-articles-output.csv'

    final_data_folder = 'data/'
    final_edu_data = final_data_folder + 'edu_all.csv'

    model_folder = 'saved_models/'
    model_electra = model_folder + 'electra/pytorch_model.bin'
    model_deberta = model_folder + 'deberta/pytorch_model.bin'

    edu_or_climate = 'climate'
    edu_or_climate = 'edu'
    data_type = 'sentence' if edu_or_climate == 'edu' else 'article'

    analysis_folder = 'codes_for_analysis/'

    outputs_folder = 'outputs/'
    exp_names = ['finetune', 'zeroshot']
    pred_file_pattern = outputs_folder + '{data_name}_{zeroshot_or_finetune}/{model}.txt'  # .format(outputs_folder, data_name=edu_or_climate, )
    file_gpt3_api = 'data/openai_key.txt'

    output_gold_input = outputs_folder + '{}_test.source'.format(edu_or_climate)
    output_gold_output = outputs_folder + '{}_test.target'.format(edu_or_climate)
    evaluation_result = analysis_folder + 'evaluation/scores_{}.json'.format(edu_or_climate)
    dev_pred_file_pattern = outputs_folder + '{data_name}_{zeroshot_or_finetune}/dev_results_to_get_threshold/{model}.txt'

    data_csv = {
        'edu': final_edu_data,
        'climate': preprocess_data_folder + "ClimateFeedbackDataset_20210825.csv"
    }
    key_sent = 'source_article'
    key_fallacy = 'logical_fallacies'
    test_size = {
        'edu': 300,
        'climate': 100,
    }
    dev_size = {
        'edu': 300,
        'climate': 10,
    }

    article_csv_raw = preprocess_data_folder + "ClimateFeedbackDataset_20210825.csv"
    article_csv_clean = preprocess_data_folder + "onehot_9class_20210825.csv"

    def __init__(self):
        self.exp_folders = {i: '{outputs_folder}{data_name}_{zeroshot_or_finetune}/'.format(
            outputs_folder=self.outputs_folder, data_name=self.edu_or_climate, zeroshot_or_finetune=i)
            for i in self.exp_names
        }

        self.save_path = {
            data_type: {split: '{}/{}_{}.csv'.format(self.final_data_folder, data_type, split)
                        for split in ['train', 'dev', 'test']}
            for data_type in self.data_csv
        }


class ScrapingFilePaths(FilePaths):
    pass


class EvaluationFilePaths(FilePaths):
    models = ['ALBERT', 'BERT', 'BigBird', 'DeBERTa', 'DistilBERT', 'ELECTRA', 'MobileBERT', 'RoBERTa', ]
    metrics = {
        'f1-score': 'MiF1',
        'Macro F1': 'MaF1',
        'precision': 'P',
        'recall': 'R',
        'exact match': 'ExMa',
    }
    key_metric = 'f1-score'


class VisualizationFilePaths(EvaluationFilePaths):
    pass


class PreprocessFilePaths(FilePaths):
    def __init__(self):
        super().__init__()
        preprocess_data_folder = self.preprocess_data_folder

        self.fallacy_list_url = 'https://en.wikipedia.org/wiki/List_of_fallacies'
        self.fallacy_list_html = preprocess_data_folder + 'fallacy_list.html'
        self.fallacy_list_csv = preprocess_data_folder + 'fallacy_list.csv'
        self.fallacy_list_txt = preprocess_data_folder + 'fallacy_list.txt'
        self.file_bad_ques_pref = preprocess_data_folder + 'sentence_bad_ques_pref.txt'
        self.file_bad_ques_suff = preprocess_data_folder + 'sentence_bad_ques_suff.txt'
        self.file_bad_defi_only = preprocess_data_folder + 'sentence_bad_defi_only.csv'
        self.file_bad_incl_terms = preprocess_data_folder + 'sentence_bad_incl_terms.txt'

        self.clean_data_csv = preprocess_data_folder + 'sentence_fallacy_data.csv'
        self.deleted_data_csv = preprocess_data_folder + 'sentence_fallacy_deleted.csv'
        self.data_csv_unaligned = preprocess_data_folder + 'sentence_fallacy_not_a_fallacy.csv'
        self.clean_data_csv_normed = preprocess_data_folder + 'sentence_fallacy_data_normed.csv'
        self.data_final_format_csv = preprocess_data_folder + 'sentence_logic_quiz.csv'
