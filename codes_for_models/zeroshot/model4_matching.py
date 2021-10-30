from efficiency.log import show_var


class EmbMatcher:
    def classification_zero_shot(self, sentence_n_labels_n_neg_labels, batch_size=10):
        # reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html

        sentence_n_labels_n_neg_labels = sorted(sentence_n_labels_n_neg_labels, key=lambda i: len(i[0].split()))
        all_sent_list, labels, _ = zip(*sentence_n_labels_n_neg_labels)

        from efficiency.function import flatten_list
        all_labels = sorted(set(flatten_list(labels)))

        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        model = AutoModel.from_pretrained('deepset/sentence_bert')

        # sentence = 'George Bush is a good communicator because he speaks effectively.'
        # labels = ['Ad Hominem', 'Post Hoc', 'Slippery Slope', 'Circular Argument']
        #
        # sent_list = [sentence]
        accuracies = []

        for start_ix in range(0, len(all_sent_list), batch_size):
            sent_list = all_sent_list[start_ix: start_ix + batch_size]
            num_sents = len(sent_list)
            from efficiency.log import show_time
            inputs = tokenizer.batch_encode_plus(all_labels + list(sent_list),
                                                 return_tensors='pt',
                                                 padding='longest')

            # run inputs through model and mean-pool over the sequence
            # dimension to get sequence-level representations
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            show_time('modeling {} seqs'.format(len(list(sent_list) + all_labels)))
            output = self._get_sent_emb(model, input_ids, attention_mask)
            show_time('Finished modeling {} seqs'.format(len(list(sent_list) + all_labels)))

            sentence_reps = output[-num_sents:].mean(dim=1)
            label_reps = output[:-num_sents].mean(dim=1)
            label2rep = dict(zip(all_labels, label_reps))

            # now find the labels with the highest cosine similarities to
            # the sentence
            import torch
            from torch.nn import functional as F
            from tqdm import tqdm

            bar = tqdm(zip(sentence_reps, sentence_n_labels_n_neg_labels), total=len(sentence_n_labels_n_neg_labels))
            from efficiency.function import avg
            for sent_rep, (sent, labels, neg_labels) in bar:
                this_all_labels = labels + neg_labels
                this_label_reps = [label2rep[i] for i in this_all_labels]
                this_label_reps = torch.stack(this_label_reps)

                similarities = F.cosine_similarity(torch.unsqueeze(sent_rep, 0), this_label_reps)
                ind = similarities.argmax()
                acc = this_all_labels[ind] in labels
                accuracies.append(acc)
                bar.set_description('accuracy mean={:.2f}%'.format(100 * avg(accuracies, decimal=4)))
                continue

                closest = similarities.argsort(descending=True)
                for ind in closest:
                    print(f'label: {this_all_labels[ind]} \t similarity: {similarities[ind]}')
                import pdb;
                pdb.set_trace()

    def _get_sent_emb(self, model, input_ids, attention_mask, batch_size=1):
        all_outputs = []
        import math
        from tqdm import tqdm
        bar = tqdm(enumerate(range(0, len(input_ids), batch_size)), total=math.ceil(len(input_ids) / batch_size),
                   desc='Calculating Sentence Embedding')
        bad_ixes = []
        for ix, start_ix in bar:
            end_ix = start_ix + batch_size
            # print(start_ix)
            # print(attention_mask[start_ix:end_ix])
            # if ix in {96, 97}:
            #     bad_ixes.append(start_ix)
            #     print('Skipping', start_ix)
            #     continue

            try:
                output = model(input_ids[start_ix:end_ix], attention_mask=attention_mask[start_ix:end_ix])[0]
            except:
                bad_ixes.append(start_ix)
            all_outputs.append(output)

        import torch
        all_embs = torch.cat(all_outputs, 0)

        show_var(['bad_ixes'])
        return all_embs


def main():
    from efficiency.function import set_seed
    set_seed(verbose=True)

    from model1_transfer_from_nli import DataReader
    dr = DataReader(data_type=['sentence', 'article'][0])
    dr.get_label_explanations()
    data = dr.sentence_n_labels_n_neg_labels

    model = EmbMatcher()

    model.classification_zero_shot(data)


if __name__ == '__main__':
    from model1_transfer_from_nli import Constants

    C = Constants()
    main()
