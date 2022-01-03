from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoConfig
load_dir="textattack/roberta-base-MNLI"
save_dir="saved_models/roberta-base-mnli"
model=AutoModelForSequenceClassification.from_pretrained(load_dir)
tokenizer = AutoTokenizer.from_pretrained(load_dir)
config = AutoConfig.from_pretrained(load_dir)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
config.save_pretrained(save_dir)