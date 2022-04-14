from transformers import AutoModelForSequenceClassification, AdamW
from logicedu import get_logger, MNLIDataset, train, eval1
import argparse
import pandas as pd
import torch
from library import eval_classwise, eval_and_store, convert_to_multilabel

torch.manual_seed(0)

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
    parser.add_argument("-s", "--savepath", help="Path to save logicclimate model")
    parser.add_argument("-mp", "--map", help="Map labels to this category")
    parser.add_argument("-ts", "--train_strat", help="Strategy number for training")
    parser.add_argument("-ds", "--dev_strat", help="Strategy number for development and testing")
    parser.add_argument("-f", "--finetune", help="Set this flag if you want to finetune the model on LogicClimate",
                        default='F')
    parser.add_argument("-c", "--classwise_savepath", help="Path to store classwise results")
    parser.add_argument("-sr", "--result_path", help="Path to store results on dev set")
    args = parser.parse_args()
    # word_bank = pickle.load('../../data/word_bank.pkl')
    logger.info(args)
    logger.info("initializing model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    logger.info("creating dataset")
    fallacy_train = pd.read_csv('../../data/climate_train_mh.csv')
    fallacy_dev = pd.read_csv('../../data/climate_dev_mh.csv')
    fallacy_test = pd.read_csv('../../data/climate_test_mh.csv')
    fallacy_train['logical_fallacies'] = fallacy_train['logical_fallacies'].apply(eval)
    fallacy_dev['logical_fallacies'] = fallacy_dev['logical_fallacies'].apply(eval)
    fallacy_test['logical_fallacies'] = fallacy_test['logical_fallacies'].apply(eval)
    if args.finetune == 'F':
        fallacy_test = pd.concat([fallacy_train, fallacy_dev, fallacy_test])
    fallacy_ds = MNLIDataset(args.tokenizer, fallacy_train, fallacy_dev, 'logical_fallacies', args.map, fallacy_test,
                             fallacy=True, train_strat=int(args.train_strat), test_strat=int(args.dev_strat),
                             multilabel=True)
    model.resize_token_embeddings(len(fallacy_ds.tokenizer))
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    if args.finetune == 'T':
        logger.info("starting training")
        train(model, fallacy_ds, optimizer, logger, args.savepath, device, ratio=1, epochs=100,
              positive_weight=float(args.weight))

        model = AutoModelForSequenceClassification.from_pretrained(args.savepath, num_labels=3)
        model.to(device)
    model.eval()
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
        logger.info(classwise_scores)
        df.to_csv(args.classwise_savepath)
    if args.result_path is not None:
        logger.info("Generating results on Dev Set")
        df = eval_and_store(fallacy_ds, model, logger, device)
        df = convert_to_multilabel(df)
        df.to_csv(args.result_path)
