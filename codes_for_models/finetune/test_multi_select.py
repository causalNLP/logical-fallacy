import shutil
import os

names = ['electra', 'albert', 'bert', 'distillbert', 'roberta', 'bigbird', 'deberta', 'mobilebert']
for i,u in enumerate(names):
    model_name = '../../results/fallacy/finetuned/' + u + '/best-epoch' 
    output_dir = '../../results/fallacy/finetuned/' + u
    length = '128'
    command = 'python main.py ' + \
    '--tokenizer_name ' + model_name + ' ' + \
    '--model_name ' + model_name + ' ' + \
    '--data_dir ../../datasets/fallacy/sentence/ ' + \
    '--max_source_length ' + length + ' ' + \
    '--max_target_length 10 ' + \
    '--val_max_target_length 10 ' + \
    '--mode output_prob ' + \
    '--output_dir ' + output_dir + ' ' + \
    '--task_type sequence_matching ' + \
    '--test_type dev ' + \
    '--overwrite_output_dir true ' + \
    '--load_best_model_at_end true ' + \
    '--logging_steps 10000000000 ' + \
    '--logging_dir ../logs ' + \
    '--fp16 true ' + \
    '--num_labels 2 ' + \
    '--save_steps 10000000000 ' + \
    '--evaluation_strategy steps ' + \
    '--num_train_epochs 5 ' + \
    '--per_device_train_batch_size 8 ' + \
    '--gradient_accumulation_steps 2 ' + \
    '--per_device_eval_batch_size 32 ' + \
    '--eval_steps 1000 ' + \
    '--save_total_limit 1 ' + \
    '--seed 42 ' + \
    '--warmup_steps 6000 ' + \
    '--learning_rate 2e-5'
    os.system(command)
    shutil.move(output_dir + '/prob.txt', "outputs/dev/fewshot_"+u+'.txt')
