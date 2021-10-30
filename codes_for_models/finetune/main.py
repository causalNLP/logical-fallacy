from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    ElectraTokenizerFast,
    Seq2SeqTrainingArguments,
    set_seed,
)
from dataclasses import dataclass, field
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.trainer_utils import EvaluationStrategy
from typing import Optional
import sys
import os
from util import *
from evaluate import *
from pathlib import Path

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_type: str = field(
        metadata={"help": "Task type, can be either generation or classification"}
    )
    num_labels: str = field(
        metadata={"help": "Number of labels, used for sequence classification"}
    )
    mode: str = field(
        metadata={"help": "mode, can be either train or test"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_type: Optional[str] = field(
        default="test", metadata={"help": "The type_path of the test file, test.seen, test.unseen etc."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=None, metadata={"help": "# training examples. None means use all."})
    n_val: Optional[int] = field(default=None, metadata={"help": "# validation examples. None means use all."})
    n_test: Optional[int] = field(default=None, metadata={"help": "# test examples. None means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

@dataclass
class EvalArguments:
    """
    Arguments pertaining to the evaluation of the model.
    """

    decode: Optional[str] = field(
        default='beam_search', metadata={"help": "Decoding method used, take in value of beam_search, nucleus"}
    )
    metric: Optional[str] = field(
        default='bleu', metadata={"help": "The metric used to evaluate the model, takes in value of bleu, rouge, meteor etc"}
    )
    compute_metric: Optional[bool] = field(
        default=False, metadata={"help": "whether to compute metrics while generating the outputs, must be False if num_samples > 1"}
    )
    num_beams: Optional[int] = field(
        default=5, metadata={"help": "beam size used to decode"}
    )
    num_samples: Optional[int] = field(
        default=1, metadata={"help": "Number of decoded sequence for each input"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[-1].endswith(".json"):
    # If the last argument ends with json then it's the path to a json file,
    # parse it to get our arguments.
        model_args, data_args, eval_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, eval_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    if model_args.task_type == 'generation':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name,
            config=config,
            cache_dir=model_args.cache_dir
        )
    
    else:
        if model_args.model_name != 'microsoft/DialogRPT-updown' and config.num_labels is None:
            config.num_labels = int(model_args.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name,
            config=config,
            cache_dir=model_args.cache_dir
        )
    
    # Get datasets
    train_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="train",
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if model_args.mode == 'train'
        else None
    )
    eval_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="dev",
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if model_args.mode == 'train' and training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path=data_args.test_type,
            task_type = model_args.task_type,
            mode = model_args.mode,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
    ) 

    # Initialize our Trainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.do_predict else eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, config.decoder_start_token_id,model_args.task_type, model_args.mode, data_args),
        tokenizer=tokenizer,
    )
   
    if model_args.mode == 'train':
        check_output_dir(training_args)#check if output_dir exists and raises error if it exists over_wirte=False

        set_seed(training_args.seed)#set training seed
        if model_args.freeze_embeds:
            freeze_embeds(model)
        if model_args.freeze_encoder:
            freeze_params(model.get_encoder())
            assert_all_frozen(model.get_encoder())
        trainer.train()
        trainer.save_model(Path(training_args.output_dir).joinpath("best-epoch"))#save best epoch

    elif model_args.mode == 'output_prob':
        out_prob(model, tokenizer, trainer.get_test_dataloader(test_dataset), Path(training_args.output_dir).joinpath("prob.txt"))

if __name__ == "__main__":
    main()
