if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device = ", device)
logger = get_logger()
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tokenizer", help="tokenizer path")
parser.add_argument("-m", "--model", help="model path")
parser.add_argument("-w", "--weight", help="Weight of entailment loss")
parser.add_argument("-s", "--savepath", help="Path to save logicedu model")
parser.add_argument("-f", "--finetune", help="Set this flag if you want to finetune the model on MNLI", default='F')
parser.add_argument("-sf", "--savepath2", help="Path to save mnli model", default="mnli")
args = parser.parse_args()
print(args)
logger.info("initializing model")
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
model.to(device)
fallacy_all=pd.read_csv('../../data/edu_all_updated.csv')[['source_article','updated_label']]
fallacy_train,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
fallacy_dev,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
fallacy_ds=MNLIDataset(args.tokenizer,fallacy_train,fallacy_dev,'updated_label',fallacy_test,fallacy=True)