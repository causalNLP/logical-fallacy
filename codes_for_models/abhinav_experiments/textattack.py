import textattack
import transformers
import argparse
from logicedu import MNLIDataset

# Load model, tokenizer, and model_wrapper
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tokenizer", help="tokenizer path")
parser.add_argument("-m", "--model", help="model path")
parser.add_argument("-mp", "--map", help="Map labels to this category")
parser.add_argument("-ts", "--train_strat", help="Strategy number for training")
parser.add_argument("-ds", "--dev_strat", help="Strategy number for development and testing")
args = parser.parse_args()
model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
fallacy_train = pd.read_csv('../../data/edu_train.csv')
fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
fallacy_test = pd.read_csv('../../data/edu_test.csv')
fallacy_ds = MNLIDataset(args.tokenizer, fallacy_train, fallacy_dev, 'updated_label', args.map, fallacy_test,
                         fallacy=True, train_strat=int(args.train_strat), test_strat=int(args.dev_strat))
eval_data = fallacy_ds.test_df
eval_data = eval_data[eval_data['gold_label'] == 'entailment']
attack = textattack.attack_recipes.input_reduction_feng_2018.InputReductionFeng2018.build(model)
for i, row in eval_data.iterrows():
    text = "<CLS> " + row['sentence1'] + " <SEP> " + row['sentence2'] + " <SEP>"
    label = 0
    result = attack.attack(text, label)
    print(result)
    break
