import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re


def add(char, num):
    i = ord(char[0])
    i += num
    char = chr(i)
    return char


def replace_char(i):
    return "[" + add('A', i) + "]"


def replace_masked_tokens(input, replace_fn=replace_char):
    j = 0
    for i in range(10):
        output = input.replace("MSK<%d>" % i, replace_fn(j))
        if input == output:
            output = input.replace("<MSK%d>" % i, replace_fn(j))
        if input == output:
            corefs = get_corefs(input)
            if corefs:
                output = input.replace(list(corefs)[0], replace_fn(j))
        if input != output:
            j += 1
        input = output
    return output


def get_corefs(text):
    return set(re.findall(r'coref.', text))


def eval_classwise(model, test_loader, logger, unique_labels, device):
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y, weights) in enumerate(test_loader):
            # logger.debug("%d", batch_idx)
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)
            if model == "random":
                prediction = torch.rand([15, 3])
            else:
                _, prediction = model(pair_token_ids,
                                      token_type_ids=seg_ids,
                                      attention_mask=mask_ids,
                                      labels=labels).values()
            all_preds.append(torch.log_softmax(prediction, dim=1).argmax(dim=1))
            all_labels.append(labels)
        all_preds = 1 - torch.stack(all_preds)
        all_labels = 1 - torch.stack(all_labels)
        all_preds[all_preds < 0] = 0
        all_labels[all_labels < 0] = 0
        results = []
        for i in range(all_preds.shape[1]):
            preds = all_preds[:, i].cpu()
            labels = all_labels[:, i].cpu()
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            no_of_p_labels = int(torch.sum(labels))
            results.append([unique_labels[i], prec, rec, f1, no_of_p_labels])
        return results


def get_label(hypothesis, ds, map='base', debug=False):
    if debug:
        print("hypo = ", hypothesis)
    for label in ds.unique_labels:
        if map == 'base' and label in hypothesis:
            return label
        elif map == 'masked-logical-form':
            form = replace_masked_tokens(list(ds.mappings[ds.mappings['Original Name'] == label]
                                              ['Masked Logical Form'])[0])
            if debug:
                print("checking %s" % form)
            if form in hypothesis:
                return label
    if debug:
        print(hypothesis)


def convert_to_multilabel(df, debug=False):
    data = []
    for text in df['text'].unique():
        selected_df = df[df['text'] == text]
        gt_labels = []
        pred_labels = []
        for i, row in selected_df.iterrows():
            if row['ground_truth'] == 'entailment':
                gt_labels.append(row['label'])
            if row['prediction'] == 'entailment':
                pred_labels.append(row['label'])
        # if debug:
        # print(gt_labels)
        # print(pred_labels)
        intersection = len(set(gt_labels) & set(pred_labels))
        if intersection == 0:
            result = "incorrect"
        elif intersection == len(gt_labels) == len(pred_labels):
            result = "exact match"
        else:
            result = "partial match"
        data.append([text, gt_labels, pred_labels, result])
    return pd.DataFrame(data, columns=['text', 'ground_truth_labels', 'model_predicted_labels', 'result'])


def eval_and_store(ds, model, logger, device, map, debug=False):
    data = []
    for i, row in ds.val_df.iterrows():
        # if i % 10 == 0:
        #     logger.debug(i)
        if debug and i == 1:
            break
        premise = row['sentence1']
        hypothesis = row['sentence2']
        label = row['gold_label']
        premise_id = ds.tokenizer.encode(premise, add_special_tokens=False)
        hypothesis_id = ds.tokenizer.encode(hypothesis, add_special_tokens=False)
        pair_token_ids = [ds.tokenizer.cls_token_id] + premise_id + [
            ds.tokenizer.sep_token_id] + hypothesis_id + [ds.tokenizer.sep_token_id]
        premise_len = len(premise_id)
        hypothesis_len = len(hypothesis_id)
        segment_ids = torch.tensor(
            [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
        attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))
        y = torch.tensor([ds.label_dict[label]]).to(device)
        _, prediction = model(torch.tensor(pair_token_ids).view(1, -1).to(device),
                              token_type_ids=segment_ids.view(1, -1).to(device),
                              attention_mask=attention_mask_ids.view(1, -1).to(device), labels=y).values()
        prediction = prediction.argmax(dim=1)
        if prediction == y:
            result = "Correct"
        else:
            result = "Wrong"
        if prediction == 0:
            prediction = "entailment"
        else:
            prediction = "contradiction"
        data.append([premise, get_label(hypothesis, ds, map, debug=debug), label, prediction, result])
    return pd.DataFrame(data, columns=['text', 'label', 'ground_truth', 'prediction', 'result'])
