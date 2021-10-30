from efficiency.log import show_var


class GPT2:
    def __init__(self):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model_id = 'gpt2-large'
        self.model = GPT2LMHeadModel.from_pretrained(model_id)  # .to('cudo')
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def classification_zero_shot(self, sentence_n_labels_n_neg_labels, data_type=['sentence', 'article'][0]):
        import random
        from tqdm import tqdm

        candidate_template = 'Please classify a piece of text into the following categories of logical fallacies: ' \
                          '{labels_str}.\n\nText: {sentence}\nLabel: {label}'
        # candidate_template = 'An example of {label} fallacy is the following: {sentence}'

        from efficiency.function import avg, set_seed
        set_seed()
        accuracies = []
        predictions = []
        bar = tqdm(list(enumerate(sentence_n_labels_n_neg_labels)))
        for sent_id, (sent, labels, neg_labels) in bar:
            all_labels = labels + neg_labels

            if '{labels_str}' in candidate_template:
                all_labels = [i.strip().capitalize() for i in all_labels]
                random.shuffle(all_labels)
                labels_str = ', '.join(all_labels)  # Post hoc, Slippery slope, Circular argument, Unknown type

            label_n_ppl = []
            for label in all_labels:
                if '{labels_str}' in candidate_template:
                    candidate = candidate_template.format(labels_str=labels_str, label=label, sentence=sent)
                else:
                    candidate = candidate_template.format(label=label, sentence=sent)
                ppl = self.seq2ppl(candidate)
                label_n_ppl.append((label.lower(), ppl))
            # show_var(['candidate'])
            # show_var(['label_n_ppl'])
            label_n_ppl = min(label_n_ppl, key=lambda i: i[-1])
            pred = label_n_ppl[0]
            # show_var(['pred'])
            # import pdb;pdb.set_trace()

            pred = pred.lower()
            preds = [pred]
            predictions.append((sent_id, preds, sent))
            if pred in {i.lower() for i in all_labels}:
                acc = pred in {i.lower() for i in labels}
                accuracies.append(acc)
                bar.set_description('accuracy mean={:.2f}%'.format(100* avg(accuracies, decimal=4)))

        import pdb;
        pdb.set_trace()
        sent2preds = {sent: preds for sent_id, preds, sent in predictions}

        return sent2preds # predictions

    def seq2ppl(self, text_input, stride=512):
        # reference: Easier perplexity computation #9648 https://github.com/huggingface/transformers/issues/9648
        # text_input = "An example of slippery slope fallacy is as follows: If I eat this donut today, I'll probably eat another donut tomorrow. If I eat one donut tomorrow, I might eat several donuts the next day."

        import torch
        max_length = self.model.config.n_positions

        encodings = self.tokenizer(text_input, return_tensors='pt')

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl


def main():
    from efficiency.function import set_seed
    set_seed(verbose=True)

    from model1_transfer_from_nli import DataReader
    dr = DataReader(data_type=['sentence', 'article'][0])
    dr.get_label_explanations()
    data = dr.sentence_n_labels_n_neg_labels


    gpt2 = GPT2()

    gpt2.classification_zero_shot(data)


if __name__ == '__main__':
    from model1_transfer_from_nli import Constants

    C = Constants()
    main()
