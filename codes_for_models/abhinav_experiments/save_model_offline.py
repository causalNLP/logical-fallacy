from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_dir")
parser.add_argument("-s", "--save_dir", help="model path")
parser.add_argument("-t", "--type", default='seq')
args = parser.parse_args()
if args.type == 'seq':
    model = AutoModelForSequenceClassification.from_pretrained(args.load_dir, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    config = AutoConfig.from_pretrained(args.load_dir)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    config.save_pretrained(args.save_dir)
elif args.type == 'st':
    model = SentenceTransformer(args.load_dir)
    model.save(args.save_dir)
else:
    print("invalid type!!")
