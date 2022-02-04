from efficiency.log import show_var
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.insert(1, '../abhinav_experiments')
from logicedu import get_logger,get_unique_labels,get_metrics
from tqdm import tqdm
import random
from sklearn.preprocessing import MultiLabelBinarizer
import openai
import time

class GPT3:
    def __init__(self):
        self.setup_gpt3()

    def setup_gpt3(self):

        # get OpenAI access key
        from efficiency.log import fread
        key = fread(C.file_gpt3_api, if_strip=True, delete_empty=True)[0]
        openai.api_key = key

        # openai.api_key = os.getenv("OPENAI_API_KEY")

    def classification_zero_shot(self, sentence_n_labels_n_neg_labels, result_file,
                                 data_type=['sentence', 'article'][0]):
        import os
        import json

        sent_id2pred_n_sent = {}
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
        import random
        from tqdm import tqdm

        if data_type == 'sentence':

            prompt_template = 'Please classify a piece of text into the following categories of logical fallacies: ' \
                              '{labels_str}.\n\nText: {sentence}\nLabel:'

            'An example of ad hominem fallacy is the following: {sentence}'
            max_tokens = 10
        else:
            prompt_template = 'Please classify a news article about climate change into the following categories of logical fallacies: ' \
                              '{labels_str}.\n\nText: {sentence}\nOne label:'
            max_tokens = 20

        from efficiency.log import fwrite
        from efficiency.function import avg, set_seed
        set_seed()
        accuracies = []
        predictions = []
        bar = tqdm(list(enumerate(sentence_n_labels_n_neg_labels)))
        for sent_id, (sent, labels, neg_labels) in bar:
            all_labels = labels + neg_labels
            all_labels = [i.strip().capitalize() for i in all_labels]

            if sent_id in sent_id2pred_n_sent:
                pred, sent_in_pred_file = sent_id2pred_n_sent[sent_id]
                if sent != sent_in_pred_file:
                    pass
                    # show_var(['sent', 'sent_in_pred_file'])
                    # import pdb;
                    # pdb.set_trace()
            else:
                random.shuffle(all_labels)
                labels_str = ', '.join(all_labels)  # Post hoc, Slippery slope, Circular argument, Unknown type
                sent_for_prompt = sent
                if len(sent.split()) > 900:
                    sent_for_prompt = ' '.join(sent.split()[:900]).rsplit('.', 1)[0] + '.'
                prompt = prompt_template.format(
                    labels_str=labels_str,
                    sentence=sent_for_prompt, )

                kwargs = dict(
                    engine="davinci",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    n=1,
                    stop="\n",
                )
                received = False
                for _ in range(10):
                    if received: break
                    try:
                        response = openai.Completion.create(**kwargs)
                        pred = response['choices'][0]['text'].strip().rstrip('.')
                        if pred in all_labels:
                            received = True
                        else:
                            kwargs['temperature'] = 0.5
                    except:
                        import sys
                        error = sys.exc_info()[0]
                        if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                            print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                            # assert False

                        print("API error:", error)
                        import time
                        time.sleep(1)
                        import pdb;pdb.set_trace()
                # show_var(['response'])
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

                # print(prompt, pred)
                # import pdb;
                # pdb.set_trace()
                fwrite(json.dumps({'sentence_id': sent_id, 'pred': pred, 'sentence': sent}) + '\n',
                       result_file, mode='a')

            preds = [pred]

            if data_type == 'article':
                preds = pred.split(', ')
            preds = [i.lower() for i in preds]
            predictions.append((sent_id, preds, sent))

            if preds[0] in {i.lower() for i in all_labels}:
                acc = {i for i in preds} & {i.lower() for i in labels}
                accuracies.append(bool(acc))
                bar.set_description('accuracy mean={:.2f}%'.format(100 * avg(accuracies, decimal=4)))
        sent2preds = {sent: preds for sent_id, preds, sent in predictions}

        return sent2preds # predictions

    def classification_few_shot(self):
        # reference: https://beta.openai.com/docs/api-reference/classifications
        import openai
        kwargs = dict(search_model="ada",
                      model="curie",
                      examples=[
                          ["A happy moment", "Positive"],
                          ["I am sad.", "Negative"],
                          ["I am feeling awesome", "Positive"]
                      ],
                      query="It is a raining day :(",
                      labels=["Positive", "Negative", "Neutral"],
                      max_examples=3,
                      return_prompt=True, )
        kwargs = dict(search_model="ada",
                      model="curie",
                      examples=[
                          # ["What can our new math teacher know? Have you seen how fat she is?", "Ad Hominem", ],
                          # ["Shortly after MySpace became popular, U.S. soldiers found Saddam Hussein.", "Post Hoc", ],
                          # [
                          #     "Major in English in college, start reading poetry, and next thing you know, you will become an unemployed pot-smoking loser.",
                          #     "Slippery Slope", ],
                          # ["George Bush is a good communicator because he speaks effectively.", "Circular Argument", ],
                      ],
                      query="Michael Jackson, Kurt Cobain, and Jimi Hendrix were rock stars who died young. Therefore, if you become a rock star, don’t expect to live a long life.",
                      labels=["Ad Hominem", "Post Hoc", "Slippery Slope", "Circular Argument", ],
                      max_examples=3,
                      return_prompt=True, )

        received = False
        from tqdm import tqdm
        for _ in tqdm(list(range(10))):
            if received: break
            try:
                response = openai.Classification.create(**kwargs)
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
        show_var(['response'])
        import pdb;
        pdb.set_trace()
        '''
        response: <OpenAIObject classification at 0x7ffcf84bc270> JSON: {
          "completion": "cmpl-3cY9glzFkmXqIgcV1IXdxj9F0Tyvd",
          "label": "Negative",
          "model": "curie:2020-05-03",
          "object": "classification",
          "prompt": "Please classify a piece of text into the following categories: Positive, Negative, Neutral.\n\nText: I am sad.\nLabel: Negative\n---\nText: A happy moment\nLabel: Positive\n---\nText: I am feeling awesome\nLabel: Positive\n---\nText: It is a raining day :(\nLabel:",
          "search_model": "ada",
          "selected_examples": [
            {
              "document": 1,
              "label": "Negative",
              "text": "I am sad."
            },
            {
              "document": 0,
              "label": "Positive",
              "text": "A happy moment"
            },
            {
              "document": 2,
              "label": "Positive",
              "text": "I am feeling awesome"
            }
          ]
        }

        response: <OpenAIObject classification at 0x7f89e027db30> JSON: {
          "completion": "cmpl-3cYNwMStEyHs9382TGImtde7gsr8Z",
          "label": "Ad hominem",
          "model": "curie:2020-05-03",
          "object": "classification",
          "prompt": "Please classify a piece of text into the following categories: Ad hominem, Post hoc, Slippery slope, Circular argument.\n\nText: Major in English in college, start reading poetry, and next thing you know, you will become an unemployed pot-smoking loser.\nLabel: Slippery slope\n---\nText: George Bush is a good communicator because he speaks effectively.\nLabel: Circular argument\n---\nText: Shortly after MySpace became popular, U.S. soldiers found Saddam Hussein.\nLabel: Post hoc\n---\nText: Michael Jackson, Kurt Cobain, and Jimi Hendrix were rock stars who died young. Therefore, if you become a rock star, don\u2019t expect to live a long life.\nLabel:",
          "search_model": "ada:2020-05-03",
          "selected_examples": [
            {
              "document": 2,
              "label": "Slippery slope",
              "object": "search_result",
              "score": 11.04,
              "text": "Major in English in college, start reading poetry, and next thing you know, you will become an unemployed pot-smoking loser."
            },
            {
              "document": 3,
              "label": "Circular argument",
              "object": "search_result",
              "score": 8.361,
              "text": "George Bush is a good communicator because he speaks effectively."
            },
            {
              "document": 1,
              "label": "Post hoc",
              "object": "search_result",
              "score": 4.823,
              "text": "Shortly after MySpace became popular, U.S. soldiers found Saddam Hussein."
            }
          ]
        }

        '''
        return response

    def classify_zero_shot2(self,data_path):
        fallacy_all=pd.read_csv(data_path)[['source_article','updated_label']]
        _,fallacy_rem=train_test_split(fallacy_all,test_size=600,random_state=10)
        _,fallacy_test=train_test_split(fallacy_rem,test_size=300,random_state=10)
        all_labels=get_unique_labels(fallacy_all,'updated_label')
        prompt_template = 'Please classify a piece of text into the following categories of logical fallacies: ' \
                              '{labels_str}.\n\nText: {sentence}\nLabel:'
        max_tokens = 10
        labels=[]
        preds=[]
        mlb = MultiLabelBinarizer()
        mlb.fit([all_labels])
        for _,row in tqdm(fallacy_test.iterrows()):
          random.shuffle(all_labels)
          labels_str = ', '.join(all_labels)  # Post hoc, Slippery slope, Circular argument, Unknown type
          sent=row['source_article']
          sent_for_prompt = sent
          if len(sent.split()) > 900:
              sent_for_prompt = ' '.join(sent.split()[:900]).rsplit('.', 1)[0] + '.'
          prompt = prompt_template.format(
              labels_str=labels_str,
              sentence=sent_for_prompt, )
          kwargs = dict(
              engine="davinci",
              prompt=prompt,
              max_tokens=max_tokens,
              temperature=0,
              n=1,
              stop="\n",
          )
          # print("Prompt is ",prompt)
          received = False
          for _ in range(10):
              if received: break
              try:
                  response = openai.Completion.create(**kwargs)
                  pred = response['choices'][0]['text'].strip().rstrip('.').lower()
                  # print("openai responded ",pred," actual is",row['updated_label'])
                  if pred in all_labels:
                      received = True
                  else:
                      kwargs['temperature'] += 0.1
              except:
                  import sys
                  error = sys.exc_info()[0]
                  if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                      print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                      # assert False
                  elif error == openai.error.RateLimitError:
                    time.sleep(10)
                  print("API error:", error)
          if received:
            preds_mh = mlb.transform([[pred]])
          else:
            preds_mh = [[0]*len(all_labels)]
          labels_mh = mlb.transform([[row['updated_label']]])
          print(pred,preds_mh)
          print(row['updated_label'],labels_mh)
          labels.append(labels_mh[0])
          preds.append(preds_mh[0])
        scores=get_metrics(preds,labels,sig=False,tensors=False)
        print("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f" %(scores[4],scores[5],scores[1],scores[2],scores[3]))


def main():
    from efficiency.function import set_seed
    set_seed(verbose=True)

    from model1_transfer_from_nli import DataReader
    data_type = ['sentence', 'article'][1]
    dr = DataReader(data_type=data_type)
    dr.get_label_explanations()
    data = dr.sentence_n_labels_n_neg_labels

    gpt3 = GPT3()
    gpt3.classification_zero_shot(data, data_type=data_type)
    # gpt3.classification_few_shot()

if __name__ == '__main__':
    from model1_transfer_from_nli import Constants

    C = Constants()
    gpt3=GPT3()
    gpt3.classify_zero_shot2('../../data/edu_all_updated.csv')
