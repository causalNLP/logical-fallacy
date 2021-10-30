def main():
    import os
    import sys
    path = os.path.abspath(os.path.join(__file__, "../../"))
    # path = os.path.abspath(os.pardir)
    sys.path.append(path)  # sys.path.append("../..")

    from file_paths import EvaluationFilePaths
    from eval_finetune_results import Results
    from efficiency.function import random_sample, set_seed

    FP = EvaluationFilePaths()
    data_type = FP.data_type
    edu_or_climate = FP.edu_or_climate

    set_seed(verbose=True)

    r = Results()
    # sent, labels, neg_labels
    raw_data = [(seq, labels, list(set(r.label_set) - set(labels)))
                for seq, labels in r.true_seq2labels.items()]

    setting2preds = {}
    for n_classes in [20, None][1:]:
        if n_classes is not None:
            data = [(sent, labels, random_sample(neg_labels, n_classes - len(labels)))
                    for sent, labels, neg_labels in raw_data]
        else:
            data = raw_data
        n_classes = len(data[0][1] + data[0][2])
        print('[Info] {}-Way Classification'.format(n_classes))

        setting2seq2preds = {}

        from codes_for_models.zeroshot.model1_transfer_from_nli import Tester
        t = Tester()
        setting2seq2preds = t.run_all_models(data, data_type=data_type, multi_label=False,
                                             model_ix=2
                                             )

        # # from codes_for_models.zeroshot.model2_gpt3 import GPT3
        # from codes_for_models.zeroshot.model3_ppl import GPT2
        #
        # # for model_name, model in [('gpt2', GPT2()), ('gpt3', GPT3())][:1]:
        # for model_name, model in [('gpt2', GPT2())]:
        #     print('[Info]' + model_name)
        #     if model_name == 'gpt3':
        #       result_file = FP.pred_file_pattern.format(
        #                 data_name=edu_or_climate, model=model_name, zeroshot_or_finetune='zeroshot')
        #       seq2preds = model.classification_zero_shot(data, result_file, data_type=data_type)
        #     else:  seq2preds = model.classification_zero_shot(data, data_type=data_type)
        #     setting2seq2preds[model_name] = seq2preds

        for setting, seq2preds in setting2seq2preds.items():
            # Optional: use real probs
            setting = f'{n_classes}way_{setting}'
            preds = []
            for _, row in r.ix2seq_n_label_n_truth.iterrows():
                seq = row['inp_seq']
                label = row['inp_label']
                ps = {p.lower() for p in seq2preds[seq]}
                preds.append(int(label in ps))
            pred_file = FP.pred_file_pattern.format(
                data_name=edu_or_climate, model=setting, zeroshot_or_finetune='zeroshot')

            from efficiency.log import fwrite
            fwrite('\n'.join(str(i) for i in preds), pred_file, verbose=True)

            setting2preds[setting] = preds
            # report = r._get_performance_report(r.ix2seq_n_label_n_truth.copy(deep=True), preds, setting)
            # print(report)
        setting2thres = {k: 0.5 for k in setting2preds}
        r.get_performance(setting2preds, setting2thres=setting2thres)


if __name__ == '__main__':
    import os
    import sys

    path = [os.path.abspath(os.pardir), os.path.abspath(os.curdir)]
    path = os.path.abspath(os.curdir)
    sys.path.append(path, )  # sys.path.append("..")

    main()
