from model1_transfer_from_nli import Constants
from efficiency.log import show_var


class DataReader:
    def __init__(self, data_type=['sentence', 'article'][0]):
        if data_type == 'sentence':
            self.read_sent_feedback()
        elif data_type == 'article':
            self.read_article_feedback()

        from efficiency.log import show_time
        show_time('Finished obtaining {} data'.format(data_type))

    def read_article_feedback(self):
        C = Constants()

        path = C.csv_article_feedback
        from efficiency.log import read_csv
        data = read_csv(path)
        header_sentence = [i for i in data[0].keys() if i.startswith('source_article')][0]
        header_labels = ['feedback_1', 'guest_comment_1']

        sentences = [i[header_sentence] for i in data]
        sentences = [i.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').strip() for i in sentences]

        urls = [i['original_url'] for i in data]
        labels = [[j for j in header_labels if i[j]][0] for i in data]

        self.sentence_n_feedback_n_ids = list(zip(sentences, labels, urls))

    def read_sent_feedback(self):
        from efficiency.log import read_csv

        from efficiency.function import set_seed
        set_seed(verbose=True)

        from efficiency.function import rstrip_word, lstrip_word
        data = read_csv(C.csv_sentence_logic)
        header_sentence = [i for i in data[0].keys() if i.startswith('Sentence')][0]
        header_label = [i for i in data[0].keys() if i.startswith('Explanations')][0]

        sentences = [i[header_sentence] for i in data]
        sentences = [i.replace('\n', ' ').strip() for i in sentences]
        labels = [i[header_label] for i in data]

        self.sentence_n_feedback_n_ids = [(s, f, i) for s, f, i in zip(sentences, labels, sentences) if f]


class GPT3:
    def __init__(self):
        self.setup_gpt3()

    def setup_gpt3(self):
        import openai

        # get OpenAI access key
        from efficiency.log import fread
        key = fread(C.file_gpt3_api, if_strip=True, delete_empty=True)[0]
        openai.api_key = key

        # openai.api_key = os.getenv("OPENAI_API_KEY")

    def generator_zero_shot(self, sentence_n_feedback_n_ids, data_type=['sentence', 'article'][0]):
        import os
        import json

        sent_id2pred_n_sent = {}
        result_file = C.file_zeroshot_gen_gpt3_pred
        if os.path.isfile(result_file):
            results = []
            with open(result_file) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            show_var(['len(results)'])
            sent_id2pred_n_sent = {i['sentence_id']: (i['pred'], i['sentence']) for i in results}

        # reference: https://beta.openai.com/docs/api-reference/classifications
        import openai
        from tqdm import tqdm

        if data_type == 'article':
            prompt_template = "A group of experts read the following article:\n{sentence}\n\n"
            prompt_template += "The experts wrote a comprehensive review of the above article:"
            stop_criterion = None
            max_tokens = 128

        else:
            prompt_template = "We should understand the logical fallacies when reading other people's claims.\n\nClaim: {sentence}\nThis claim is problematic because"
            stop_criterion = "Claim"
            max_tokens = 50

        '''
        We should understand the logical fallacies when reading other people's claims.

        Claim: You think labor unions are good? You know who else liked labor unions? Karl Marx, that’s who.
        This claim is problematic because it is a non sequitur. It does not follow that because Karl Marx liked labor unions, that labor unions are bad.

        Claim: You think the minimum wage is good? You know
        '''

        from efficiency.log import fwrite
        from efficiency.function import avg, set_seed
        set_seed()
        accuracies = []
        predictions = []
        bar = tqdm(sentence_n_feedback_n_ids)
        for sent, feedback, sent_id in bar:
            if sent_id in sent_id2pred_n_sent:
                pred, sent_in_pred_file = sent_id2pred_n_sent[sent_id]
                if sent != sent_in_pred_file:
                    show_var(['sent', 'sent_in_pred_file'])
                    import pdb;
                    pdb.set_trace()
            else:
                prompt = prompt_template.format(
                    sentence=sent, )

                kwargs = dict(
                    engine="davinci",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    n=1,
                    frequency_penalty=0.5,
                    stop=stop_criterion,
                )

                received = False
                for _ in range(10):
                    if received: break
                    try:
                        response = openai.Completion.create(**kwargs)
                        received = True
                    except:
                        import sys
                        error = sys.exc_info()[0]
                        if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                            print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                            # assert False

                        print("API error:", error)
                        import time
                        time.sleep(1)
                '''
                response: <OpenAIObject text_completion id=cmpl-3cZRMI6OLT4TT57LXyNnxiu9b99D1 at 0x2b70f882a270> JSON: {                               | 1/10 [00:01<00:13,  1.48s/it]
                  "choices": [
                    {
                      "finish_reason": "stop",
                      "index": 0,
                      "logprobs": null,
                      "text": " False inductions"
                    }
                  ],
                  "created": 1630321708,
                  "id": "cmpl-3cZRMI6OLT4TT57LXyNnxiu9b99D1",
                  "model": "davinci:2020-05-03",
                  "object": "text_completion"
                }
                '''
                pred = response['choices'][0]['text'].strip()
                show_var(['response'])
                print(prompt, pred)
                import pdb;
                pdb.set_trace()
                fwrite(json.dumps(
                    {'sentence_id': sent_id, 'pred': pred[0].capitalize() + pred[1:], 'human': feedback, 'sentence': sent, }) + '\n',
                       result_file, mode='a')


        import pdb;
        pdb.set_trace()


def main():
    from efficiency.function import set_seed
    set_seed(verbose=True)

    data_type = ['sentence', 'article'][1]

    dr = DataReader(data_type=data_type)
    data = dr.sentence_n_feedback_n_ids

    gpt3 = GPT3()
    gpt3.generator_zero_shot(data, data_type=data_type)
    # gpt3.classification_few_shot()


if __name__ == '__main__':
    C = Constants()
    main()
