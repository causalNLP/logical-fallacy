import pickle
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification
import pandas as pd
import random
from torch import nn
import sklearn
import numpy as np
import sklearn.metrics
import logging
import argparse
from random import sample
from library import eval_classwise, eval_and_store, convert_to_multilabel, get_corefs, replace_masked_tokens, \
    replace_char
from weighted_cross_entropy import CrossEntropyLoss

torch.manual_seed(0)


def get_unique_labels(df, label_col_name, multilabel=False):
    labels_dict = {}
    if multilabel:
        for labels in df[label_col_name]:
            for label in labels:
                if label not in labels_dict.keys():
                    labels_dict[label] = 0
                labels_dict[label] += 1
    else:
        candidate_labels = df[label_col_name].unique()
        for label in candidate_labels:
            labels = label.split(';')
            if len(labels) > 1:
                print(labels)
            labels = [x.strip() for x in labels]
            for lbl in labels:
                if lbl not in labels_dict.keys():
                    labels_dict[lbl] = 0
                labels_dict[lbl] += 1
    return list(labels_dict.keys()), labels_dict


def replace_random_sample(i):
    return sample(word_bank, 1)[0]


class MNLIDataset:

    def __init__(self, tokenizer_path, train_df, val_df, label_col_name, map='base', test_df=None, fallacy=False,
                 undersample_train=False, undersample_val=False, undersample_test=False, undersample_rate=0.04,
                 train_strat=1, test_strat=1, multilabel=False):
        # strat is used in convert_to_mnli function
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        torch.manual_seed(0)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        special_tokens_dict = {
            'additional_special_tokens': ["[A]", "[B]", "[C]", "[D]", "[E]", "[F]", "[G]", "[H]", "[I]"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.train_data = None
        self.val_data = None
        self.label_col_name = label_col_name
        self.map = map
        self.mappings = pd.read_csv("../../data/mappings.csv")
        self.multilabel = multilabel
        if fallacy:
            self.unique_labels, self.counts_dict = get_unique_labels(pd.concat([train_df, val_df, test_df]),
                                                                     self.label_col_name,
                                                                     multilabel)
            self.total_count = 0
            for count in self.counts_dict.values():
                self.total_count += count
        self.init_data(fallacy, undersample_train, undersample_val, undersample_test,
                       undersample_ratio=undersample_rate, train_strat=train_strat, test_strat=test_strat)

    def convert_to_mnli(self, df, undersample=False, undersampling_rate=0.02, strat=1):
        """
        strat=1 -> only original article
        strat=2 -> only masked article
        strat=3 -> both
        """
        data = []
        for i, row in df.iterrows():
            for label in self.unique_labels:
                entry = [row['source_article']]
                if self.map == 'base':
                    entry.append("This is an example of %s logical fallacy" % label)
                elif self.map == 'simplify':
                    simplified_label = \
                        list(self.mappings[self.mappings['Original Name'] == label]['Understandable Name'])[0]
                    entry.append("This is an example of %s" % simplified_label)
                elif self.map == 'description':
                    description = list(self.mappings[self.mappings['Original Name'] == label]['Description'])[0]
                    entry.append("This is an example of %s" % description)
                elif self.map == 'logical-form':
                    form = list(self.mappings[self.mappings['Original Name'] == label]['Logical Form'])[0]
                    entry.append("This article matches the following logical form: %s" % form)
                elif self.map == 'masked-logical-form':
                    # print(self.mappings[self.mappings['Original Name'] == label]['Masked Logical Form'])
                    form = replace_masked_tokens(list(self.mappings[self.mappings['Original Name'] == label]
                                                      ['Masked Logical Form'])[0])
                    entry.append("This article matches the following logical form: %s" % form)
                if (self.multilabel is False and label == row[self.label_col_name]) or (
                        self.multilabel is True and label in row[self.label_col_name]):
                    entry.append("entailment")
                else:
                    p = random.random()
                    if undersample and p > undersampling_rate:
                        continue
                    entry.append("contradiction")
                weight = (self.total_count / self.counts_dict[label]) / 10
                if entry[-1] == "entailment":
                    weight *= 12
                entry.append(weight)
                # print(entry)
                entry.append(label)
                if strat % 2:
                    data.append(entry)
                if strat > 1:
                    entry1 = [replace_masked_tokens(row['masked_articles']), entry[1], entry[2], entry[3]]
                    if entry1[0] != entry[0] or strat == 3:
                        data.append(entry1)

        return pd.DataFrame(data, columns=['sentence1', 'sentence2', 'gold_label', 'weight','logical_fallacy'])

    def init_data(self, fallacy, undersample_train=False, undersample_val=False, undersample_test=False,
                  undersample_ratio=0.02, train_strat=1, test_strat=1):
        if fallacy:
            self.train_df = self.convert_to_mnli(self.train_df, undersample=undersample_train,
                                                 undersampling_rate=undersample_ratio, strat=train_strat)
            self.val_df = self.convert_to_mnli(self.val_df, undersample=undersample_val,
                                               undersampling_rate=undersample_ratio, strat=test_strat)
            self.test_df = self.convert_to_mnli(self.test_df, undersample=undersample_test,
                                                undersampling_rate=undersample_ratio, strat=test_strat)
        # print(self.counts_dict, self.total_count)
        # self.train_df.to_csv('results/temp.csv')
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
        self.test_data = self.load_data(self.test_df)

    def load_data(self, df):
        if df is None:
            return None

        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []
        weights = []

        df = df.dropna()

        premise_list = df['sentence1'].to_list()
        hypothesis_list = df['sentence2'].to_list()
        label_list = df['gold_label'].to_list()
        weight_list = df['weight'].to_list()

        for (premise, hypothesis, label, weight) in zip(premise_list, hypothesis_list, label_list, weight_list):

            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [
                self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
            # pair_token_ids = premise_id + hypothesis_id
            # print("max token id=", max(pair_token_ids))
            premise_len = len(premise_id)
            hypothesis_len = len(hypothesis_id)

            segment_ids = torch.tensor(
                [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values
            if len(pair_token_ids) > MAX_LEN or len(segment_ids) > MAX_LEN or len(attention_mask_ids) > MAX_LEN:
                continue
            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            y.append(self.label_dict[label])
            weights.append(weight)

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        # print(token_ids.shape)
        # print(torch.tensor(weights).shape)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y, torch.tensor(weights))
        # print(len(dataset))
        return dataset

    def get_undersampled(self, ratio=0.04, batch_size=32, shuffle=True):
        train_df_entail = self.train_df[self.train_df.gold_label == 'entailment']
        train_df_contradict = self.train_df[self.train_df.gold_label == 'contradiction']
        train_df_contradict = train_df_contradict.sample(n=int(len(train_df_contradict) * ratio))
        df = pd.concat([train_df_contradict, train_df_entail])
        df = self.load_data(df)
        return DataLoader(
            df,
            shuffle=shuffle,
            batch_size=batch_size
        )

    def get_data_loaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=shuffle,
            batch_size=batch_size
        )
        if self.test_data:
            test_loader = DataLoader(
                self.test_data,
                shuffle=False,
                batch_size=len(self.unique_labels)
            )
        else:
            test_loader = None

        return train_loader, val_loader, test_loader


def get_logger(level='DEBUG'):
    logger = logging.getLogger()
    logger.handlers = []
    file_log_handler = logging.FileHandler('logfile.log')
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    logger.setLevel(level)
    return logger


def multi_acc(y_pred, y_test, flip=True):
    preds = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    acc = (preds == y_test).sum().float() / float(y_test.size(0))
    if flip:
        y_pred = 1 - preds.cpu()
        y_test = 1 - y_test.cpu()
    else:
        y_pred = preds.cpu()
        y_test = y_test.cpu()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_test[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_test[i] != y_pred[i]:
            FP += 1
        if y_test[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_test[i] != y_pred[i]:
            FN += 1
    if TP == 0:
        return acc, 0, 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return acc, precision, recall


import time


def get_metrics(logits, labels, threshold=0.5, sig=True, tensors=True):
    if sig:
        sig = nn.Sigmoid()
        preds = sig(logits)
        preds = (preds > threshold)
    else:
        preds = logits
    # print(preds.cpu().int(),labels.cpu().int())
    if tensors:
        y_true = labels.cpu().int()
        y_pred = preds.cpu().int()
    else:
        y_true = np.array(labels)
        y_pred = np.array(preds)
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    accuracy = temp / y_true.shape[0]
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    exact_match = sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    micro_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, exact_match, micro_f1_score, macro_f1_score


def train(model, dataset, optimizer, logger, save_path, device, epochs=5, ratio=0.04, positive_weight=12, debug=False):
    train_loader, val_loader, _ = dataset.get_data_loaders()
    min_val_loss = float('inf')
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss_fn.to(device)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        total_train_prec = 0
        total_train_rec = 0
        if ratio < 1:
            train_loader = dataset.get_undersampled(ratio)
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y, weights) in enumerate(train_loader):
            # print(weights.shape)
            if batch_idx % 10 == 0:
                logger.debug('%d %d', epoch, batch_idx)
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)
            weights = weights.to(device)
            if debug:
                print(pair_token_ids.shape, seg_ids.shape, mask_ids.shape)
                break
            _, prediction = model(pair_token_ids,
                                  token_type_ids=seg_ids,
                                  attention_mask=mask_ids,
                                  labels=labels).values()
            # print(weights.shape)
            loss = loss_fn(prediction, labels)
            # print(loss.shape)
            len1 = loss.shape[0]
            loss = torch.dot(loss, weights) / len1
            # print('forward prop done')
            acc, prec, rec = multi_acc(prediction, labels)
            # print(acc,prec,rec)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.float()
            total_train_prec += prec
            total_train_rec += rec
        if debug:
            break
        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        train_prec = total_train_prec / len(train_loader)
        train_rec = total_train_rec / len(train_loader)
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        total_val_prec = 0
        total_val_rec = 0

        with torch.no_grad():
            for batch_idx, (pair_token_ids, mask_ids, seg_ids, y, weights) in enumerate(val_loader):
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)
                weights = weights.to(device)
                _, prediction = model(pair_token_ids,
                                      token_type_ids=seg_ids,
                                      attention_mask=mask_ids,
                                      labels=labels).values()
                loss = loss_fn(prediction, labels)
                len1 = loss.shape[0]
                loss = torch.dot(loss, weights) / len1
                acc, prec, rec = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.float()
                total_val_prec += prec
                total_val_rec += rec

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        val_rec = total_val_rec / len(val_loader)
        val_prec = total_val_prec / len(val_loader)
        flag = True
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            logger.info("saving model")
            model.save_pretrained(save_path)
        else:
            flag = False
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        logger.info(
            f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_prec: {train_prec:.4f} '
            f'train_rec: {train_rec:.4f}| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_prec: {val_prec:.4f} '
            f'val_rec: {val_rec:.4f}'
        )
        logger.info("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        if not flag:
            break


def eval1(model, test_loader, logger, device):
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y, weights) in enumerate(test_loader):
            logger.debug("%d", batch_idx)
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
        return get_metrics(all_preds, all_labels, sig=False)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger = get_logger()
    logger.info("device = %s", device)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser.add_argument("-m", "--model", help="model path")
    parser.add_argument("-w", "--weight", help="Weight of entailment loss")
    parser.add_argument("-s", "--savepath", help="Path to save logicedu model")
    parser.add_argument("-f", "--finetune", help="Set this flag if you want to finetune the model on MNLI", default='F')
    parser.add_argument("-sf", "--savepath2", help="Path to save mnli model", default="mnli")
    parser.add_argument("-mp", "--map", help="Map labels to this category")
    parser.add_argument("-ts", "--train_strat", help="Strategy number for training")
    parser.add_argument("-ds", "--dev_strat", help="Strategy number for development and testing")
    parser.add_argument("-rs", "--replace_strat", help="Function used to replace masked words", default='char')
    parser.add_argument("-rc", "--replace_count", help="Number of times to replace masked words", default=1)
    parser.add_argument("-np", "--multinli_path", help="Path for multinli directory", default="")
    parser.add_argument("-do", "--downsample", help="Downsample Training Set", default='F')
    parser.add_argument("-c", "--classwise_savepath", help="Path to store classwise results")
    parser.add_argument("-sr", "--result_path", help="Path to store results on dev set")
    parser.add_argument("-nt", "--do_not_train", help="Set this to T if you do not wish to train the model",
                        default='F')
    args = parser.parse_args()
    # word_bank = pickle.load('../../data/word_bank.pkl')
    logger.info(args)
    logger.info("initializing model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)

    if args.finetune == 'T':
        logger.info("initializing mnli dataset")
        mnli_train = pd.read_csv(args.multinli_path + "/multinli_1.0_train.txt", sep='\t',
                                 skiprows=[24809, 33960, 75910, 100113, 150637, 158833, 173103, 178251, 221950, 286844,
                                           314109])[['gold_label', 'sentence1', 'sentence2']]
        mnli_train = mnli_train.dropna()
        mnli_dev_matched = pd.read_csv(args.multinli_path + "/multinli_1.0_dev_matched.txt", sep='\t')[
            ['gold_label', 'sentence1', 'sentence2']]
        mnli_dev_mismatched = pd.read_csv(args.multinli_path + "/multinli_1.0_dev_mismatched.txt", sep='\t')[
            ['gold_label', 'sentence1', 'sentence2']]
        mnli_dev = pd.concat([mnli_dev_matched, mnli_dev_mismatched], ignore_index=True, sort=False)
        mnli_dev = mnli_dev.dropna()
        mnli_dev = mnli_dev[mnli_dev.gold_label != '-']
        mnli_ds = MNLIDataset(args.tokenizer, mnli_train, mnli_dev, 'gold_label', fallacy=False)
        logger.info("finetune on mnli")
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        train(model, mnli_ds, optimizer, logger, args.savepath2, ratio=1, epochs=10, positive_weight=1)
        logger.info("reinit model")
        model = AutoModelForSequenceClassification.from_pretrained(args.savepath2)
        model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    logger.info("creating dataset")
    fallacy_train = pd.read_csv('../../data/edu_train.csv')
    if args.downsample == 'T':
        fallacy_train = fallacy_train[:747]
    fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
    fallacy_test = pd.read_csv('../../data/edu_test.csv')
    fallacy_ds = MNLIDataset(args.tokenizer, fallacy_train, fallacy_dev, 'updated_label', args.map, fallacy_test,
                             fallacy=True, train_strat=int(args.train_strat), test_strat=int(args.dev_strat))
    model.resize_token_embeddings(len(fallacy_ds.tokenizer))
    fallacy_ds.train_df.to_csv('processed_train_df.csv')
    fallacy_ds.val_df.to_csv('processed_val_df.csv')
    # print(fallacy_ds.tokenizer.tokenize("[A] causes [B]"))
    # print("checking length", len(fallacy_ds.tokenizer), model.config.vocab_size)
    if args.do_not_train == 'F':
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        logger.info("starting training")
        train(model, fallacy_ds, optimizer, logger, args.savepath, device, ratio=1, epochs=10,
              positive_weight=int(args.weight))

        model = AutoModelForSequenceClassification.from_pretrained(args.savepath, num_labels=3)
        model.to(device)
    logger.info("starting testing")
    _, _, test_loader = fallacy_ds.get_data_loaders()
    scores = eval1(model, test_loader, logger, device)
    logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f", scores[4], scores[5], scores[1],
                scores[2], scores[3])
    if args.classwise_savepath is not None:
        classwise_scores = eval_classwise(model, test_loader, logger, fallacy_ds.unique_labels, device)
        df = pd.DataFrame.from_records(classwise_scores, columns=['Fallacy Name', 'Precision', 'Recall', 'F1',
                                                                  'Number of Positive Labels for this Class in Test Set'
                                                                  ])
        df.to_csv(args.classwise_savepath)
    if args.result_path is not None:
        logger.info("Generating results on Dev Set")
        df = eval_and_store(fallacy_ds, model, logger, device, args.map)
        df = convert_to_multilabel(df)
        df.to_csv(args.result_path)
