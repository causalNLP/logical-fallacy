import os
import sys

path = os.path.abspath(os.path.join(__file__, "../../../"))
# path = os.path.abspath(os.pardir)
sys.path.append(path)  # sys.path.append("../..")

from file_paths import EvaluationFilePaths

FP = EvaluationFilePaths()


class Results:
    def __init__(self, ):
        self.true_seq2labels, self.ix2label, self.ix2seq_n_label_n_truth, self.label_set = \
            self.get_ground_truth()

    def get_ground_truth(self, ):
        from collections import defaultdict, OrderedDict
        from efficiency.log import fread
        inputs = fread(FP.output_gold_input, if_strip=True)
        outputs = fread(FP.output_gold_output, if_strip=True)

        ix2label = []
        ix2seq_n_label_n_truth = []
        seq2labels = defaultdict(list)
        label_set = OrderedDict()
        for inp, out in zip(inputs, outputs):
            # For NLI models, the label "2" is entailment (i.e., our input label is true),
            # and "0" is contradiction (i.e., our input label is false)
            label_is_true = int(int(out) == 2)

            'the text has the logical fallacy of cum hoc ergo propter hoc </s> since lucy sutton became vice president of the parent-teacher association, student performance has declined and teacher morale is down. we on the school board believe that sutton bears sole responsibility for the downtrend.'
            inp_label, inp_seq = inp.split('</s>', 1)
            inp_label = inp_label.split('the text has the logical fallacy of', 1)[-1].strip()

            label_set[inp_label] = None
            ix2label.append({'inp_label': inp_label, 'label_is_true': label_is_true})
            ix2seq_n_label_n_truth.append({"inp_seq": inp_seq, "inp_label": inp_label, "label_is_true": label_is_true})
            if label_is_true:
                ground_truth = inp_label
                seq2labels[inp_seq].append(ground_truth)

        import pandas as pd
        ix2label = pd.DataFrame(ix2label)
        ix2seq_n_label_n_truth = pd.DataFrame(ix2seq_n_label_n_truth)

        return seq2labels, ix2label, ix2seq_n_label_n_truth, label_set

    def load_results(self):
        import os
        from efficiency.log import fread

        setting2preds = {}
        setting2dev_preds = {}

        for exp_name, exp_folder in list(FP.exp_folders.items()):
            files = [i for i in os.listdir(exp_folder) if i.endswith('.txt')]
            files = sorted(files)
            for file in files:
                model_name = file.rsplit('.txt', 1)[0]
                dev_pred_file = FP.dev_pred_file_pattern.format(
                    data_name='edu', model=model_name, zeroshot_or_finetune='finetune')
                # dev_pred_file = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1ej1clS2Y-GQdrZZDEzCK4EXi20VA76Hq/outputs/dev/fewshot_{}.txt'.format(model_name)
                # dev_pred_file = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1ej1clS2Y-GQdrZZDEzCK4EXi20VA76Hq/outputs/fewshot_sentence/{}.txt'.format(model_name)

                preds = fread(exp_folder + file, if_strip=True)
                preds = [float(i) for i in preds]

                setting = exp_name + '_' + model_name.replace('distillbert', 'distilbert')

                # if setting not in {
                #     'finetune_deberta',
                #     'finetune_electra'}:
                #     continue
                setting2preds[setting] = preds

                if os.path.isfile(dev_pred_file):
                    dev_preds = fread(dev_pred_file, if_strip=True)
                    dev_preds = [float(i) for i in dev_preds]
                    setting2dev_preds[setting] = dev_preds
        return setting2preds, setting2dev_preds

    def _perf_df_to_best_thres(self, df):
        df = df.copy(deep=True)
        df['setting'] = df.index
        df['model'] = df.apply(lambda row: row.setting.rsplit('_', 1)[0], axis=1)
        df['thres'] = df.apply(lambda row: float(row.setting.rsplit('_', 1)[-1]), axis=1)
        df_sorted = df.sort_values(FP.key_metric, ascending=False)
        df = df.sort_values(FP.key_metric, ascending=False).drop_duplicates(['model'])
        # setting = df[FP.key_metric].idxmax()
        model_list = df['model'].to_list()
        thres_list = df['thres'].to_list()
        setting2thres = dict(zip(model_list, thres_list))
        print(df_sorted)
        print('[Info] Best setting2thres decided based on dev set:', setting2thres)
        import pdb;
        pdb.set_trace()
        df_sorted.to_json(FP.evaluation_result)
        return setting2thres

    def get_performance(self, setting2preds, setting2thres=None, print_result=True, print_class_report=False,
                        thres_interval=
                        0.5
                        # 0.05
                        ):
        import numpy as np
        from tqdm import tqdm
        from collections import defaultdict

        total_progress = len(setting2preds)

        if setting2thres is not None:
            setting2thres_list = defaultdict(lambda: [0.5])
            setting2thres_list.update({k: [v] for k, v in setting2thres.items()})
        else:
            thres_range = list(np.arange(0, 1 + thres_interval, thres_interval))
            setting2thres_list = defaultdict(lambda: thres_range)
            total_progress *= len(thres_range)

        setting2performance = {}
        with tqdm(total=total_progress, disable=(total_progress < 20)) as bar:
            for setting, preds in setting2preds.items():
                df = self.ix2seq_n_label_n_truth

                thres_range = setting2thres_list[setting]
                for thres in thres_range:
                    bar.update(1)

                    report = self._get_performance_report(df.copy(deep=True), preds, setting, thres=thres,
                                                          print_class_report=print_class_report)
                    setting2performance[f'{setting}_{thres:.2f}'] = report

        import pandas as pd
        perf_df = pd.DataFrame(setting2performance).T
        # perf_df.sort_values('f1-score', ascending=False, inplace=True)
        perf_df *= 100
        # print(perf_df)

        # import pdb;
        # pdb.set_trace()
        if print_result:
            perf_df.to_json(FP.evaluation_result)
            self._perf_df_to_latex_table(perf_df)
        return perf_df

    def _get_performance_report(self, df, preds, setting, thres=0.5, decimal=4, print_class_report=False):

        import numpy as np
        import pandas as pd

        df[setting] = preds

        col_correct = setting + '_correct'
        # when label_is_true == 1, we want the prob to be \in (0.5, 1]
        # when label_is_true == 0, we want the prob to be \in [0, 0.5)
        check_correct = lambda i: pd.Series({col_correct:
                                                 np.abs(i['label_is_true'] - i[setting]) <= thres
                                             })
        correctness = df.apply(check_correct, axis=1)

        df_merged = pd.concat([df, correctness], axis=1)

        # grouped_df[col_correct].sum()
        grouped_df = df_merged.groupby(['inp_seq'])
        grouped_df.size()
        corr_sum = grouped_df.agg({col_correct: sum})
        corr_all = grouped_df.agg({col_correct: all})
        acc = round(corr_all.sum() / len(corr_all), decimal)

        report = {
            # 'accuracy by test sample': float(acc),
            # 'accuracy by test NLI': avg(correctness, decimal=decimal),
        }
        sklearn_report = self._get_sklearn_report(grouped_df, setting, thres=thres,
                                                  verbose=print_class_report)
        report.update(sklearn_report)

        return report

    def _get_sklearn_report(self, grouped_df, setting, thres=0.5, short=True, verbose=False):
        import pandas as pd

        label_is_true_arrays = []
        pred_arrays = []
        for key, item in grouped_df:
            a_group = grouped_df.get_group(key).set_index('inp_label')
            label_is_true_arrays.append(a_group['label_is_true'])
            pred_arrays.append(a_group[setting])
        # import json
        # print(json.dumps(report, indent=4))

        pred_arrays = pd.concat(pred_arrays, axis=1).fillna(0)
        label_is_true_arrays = pd.concat(label_is_true_arrays, axis=1).fillna(0)
        THRESHOLD, upper, lower = thres, 1, 0

        # y_pred = np.where(pred_arrays > THRESHOLD, upper, lower)
        pred_arrays = pred_arrays > THRESHOLD
        from sklearn.metrics import classification_report
        y_true = label_is_true_arrays.T
        y_pred = pred_arrays.T
        sklearn_report = classification_report(
            y_true,  # binary matrix of "n_samples * n_class"
            y_pred,
            target_names=y_pred.columns,
            zero_division=0,
            digits=4,
            output_dict=True,
        )

        report_df = pd.DataFrame(sklearn_report).T
        report = report_df[-4:].to_string()
        if short:
            from sklearn.metrics import hamming_loss, accuracy_score

            report = {'exact match': accuracy_score(y_true, y_pred),
                      # 'hamming loss': hamming_loss(y_true, y_pred),
                      'Macro F1': sklearn_report['macro avg'][FP.key_metric],
                      }
            report.update(sklearn_report['micro avg'])  # 'f1-score'

        if verbose:
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(report_df)
            # print(report_df.sort_values('support', ascending=False).to_latex())
            report_df['support'] /= sklearn_report['micro avg']['support']

            select_columns = ['f1-score', 'precision', 'recall', 'support']
            rename_key_map = {'f1-score': 'F1', 'precision': 'P', 'recall': 'R', 'support': 'Data\%'}
            latex = Formatter.df_to_latex(report_df, select_columns=select_columns, rename_key_map=rename_key_map,
                                          sort_by='support', task_name=setting)

            import pdb;
            pdb.set_trace()
        return report

    def _perf_df_to_latex_table(self, df):
        metrics = FP.metrics

        select_columns = metrics.keys()
        rename_key_map = metrics
        sort_by = FP.key_metric

        finetune = any('finetune_' in i for i in df.index)
        if not finetune:
            task_name = 'zeroshot'
            latex = Formatter.df_to_latex(df, select_columns=select_columns, rename_key_map=rename_key_map,
                                          sort_by=sort_by, task_name=task_name)
        else:
            raw_df = df
            for task_name in FP.exp_folders:
                print('[Info] Generating latex for task:', task_name)
                df = raw_df.filter(regex='^' + task_name + '_', axis=0)

                if task_name == 'zeroshot':
                    latex = Formatter.df_to_latex(df, select_columns=select_columns, rename_key_map=rename_key_map,
                                              sort_by=sort_by, task_name=task_name)
                    import pdb;pdb.set_trace()
                    continue

                for i in df.index:
                    model_lower = i.split('_')[1]
                    if model_lower not in {m.lower() for m in FP.models}:
                        import pdb;
                        pdb.set_trace()
                rename_index_map = {i: [m for m in FP.models if m.lower() == i.split('_')[1]][0]
                                    for i in df.index}
                latex = Formatter.df_to_latex(df, select_columns=select_columns, rename_key_map=rename_key_map,
                                              rename_index_map=rename_index_map, sort_by=sort_by, task_name=task_name)


class Formatter:
    @staticmethod
    def df_to_latex(df, select_columns=None, remove_columns=[], rename_key_map={}, rename_index_map={},
                    sort_by=None, task_name='exp', auto_scale=True, need_title_case=True, verbose=True):
        if select_columns is None:
            select_columns = df.columns
        df = df[select_columns]
        df.drop(remove_columns, axis=1)
        if sort_by:
            df.sort_values(sort_by, ascending=False, inplace=True)

        if rename_key_map:
            df.rename(columns=rename_key_map, inplace=True)
        else:
            df.columns = [i.title() for i in df.columns]

        if rename_index_map:
            df.rename(index=rename_index_map, inplace=True)
        else:
            df.index = [i.title() for i in df.index]

        if auto_scale:
            for col in df.columns:
                try:
                    if df[col].min() >= 0 and df[col].max() <= 1:
                        df[col] *= 100
                except:
                    pass

        latex = df.to_latex(bold_rows=False, float_format="%.2f", position='ht',
                            column_format='l' + 'c' * len(select_columns),
                            label='tab:' + task_name, caption='Model performance on {}'.format(task_name))
        latex = latex.replace('\centering', '\centering\small')
        if verbose: print(df); print(latex)
        return latex


def main():
    r = Results()

    setting2preds, setting2dev_preds = r.load_results()
    dev_perf_df = r.get_performance(setting2dev_preds, print_result=False)
    setting2thres = r._perf_df_to_best_thres(dev_perf_df)
    r.get_performance(setting2preds, setting2thres=setting2thres)

    electra_setting2preds = {s: p for s, p in setting2preds.items() if 'electra' in s.lower()}
    from efficiency.log import show_var
    show_var(['electra_setting2preds.keys()'])
    import pdb;pdb.set_trace()
    r.get_performance(electra_setting2preds,
                      setting2thres=setting2thres, print_class_report=True)

    import pandas as pd
    df = pd.read_json(FP.evaluation_result)
    r._perf_df_to_latex_table(df)


if __name__ == '__main__':
    main()
