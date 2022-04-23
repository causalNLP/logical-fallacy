import argparse
from logicedu import get_logger
import stanza
import spacy
import string
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import pandas as pd
import pickle
from stanza.server import CoreNLPClient
import re
from library import get_corefs


class Node:
    def __init__(self):
        self.edges = []
        self.marked_range = []

    def insert(self, range):
        self.marked_range.append(range)


class Edge:
    def __init__(self, input_index, input_range, output_index, output_range, weight):
        self.weight = weight
        self.output_range = output_range
        self.output_index = output_index
        self.input_range = input_range
        self.input_index = input_index


def is_punctuation(x):
    if len(x) == 1 and x in string.punctuation:
        return True
    return False


def insert(phrase, edges, nodes):
    i = phrase[1][1][0]
    j = phrase[2][1][0]
    rangei = (phrase[1][1][1], phrase[1][1][2])
    rangej = (phrase[2][1][1], phrase[2][1][2])
    edge1 = Edge(i, rangei, j, rangej, phrase[0])
    edge2 = Edge(j, rangej, i, rangei, phrase[0])
    edges.append(edge1)
    # print(i, len(nodes))
    nodes[i].edges.append(edge1)
    nodes[j].edges.append(edge2)


def overlap(new_range, marked_range):
    for range in marked_range:
        if (range[1] > new_range[0] >= range[0]) or (range[0] < new_range[1] <= range[1]) or (
                new_range[0] < range[0] and new_range[1] > range[1]):
            return True
    return False


def visit(i, range, component, nodes):
    if overlap(range, nodes[i].marked_range):
        return
    nodes[i].insert(range)
    component.append((i, range))
    for edge in nodes[i].edges:
        if edge.input_range == range:
            visit(edge.output_index, edge.output_range, component, nodes)


def get_connected_component(edge, component, nodes):
    visit(edge.input_index, edge.input_range, component, nodes)


def get_component_index(phrase_count, phrase_size_count, connected_components):
    is_start = False
    for i in range(len(connected_components)):
        for node in connected_components[i]:
            if phrase_count == node[0] and node[1][0] <= phrase_size_count < node[1][1]:
                if phrase_size_count == node[1][0]:
                    is_start = True
                return i, is_start
    return None, is_start


def remove_repeated_subphrases(text, next_token_dict, prev_token_dict):
    for key, value in next_token_dict.items():
        # print(key, value)
        if len(value) == 1 and "MSK" in list(value)[0] and len(prev_token_dict[int(list(value)[0][4])]) == 1:
            text = text.replace("MSK<%d> " % key, "")
    return text


def mask_out_content(text, model, client, debug=False):
    try:
        ann = client.annotate(text)
        if debug:
            print(ann.corefChain)
        text = get_coref(text, ann, debug=debug)
        if debug:
            print(text)
        ann = client.annotate(text)
        phrases = []
        curr = []
        # print("printing tokens")
        for sent in ann.sentence:
            for token in sent.token:

                if token.lemma.lower() not in sw_spacy and not is_punctuation(token.word):
                    curr.append(token.lemma.lower())
                elif len(curr) > 0:
                    phrases.append(curr)
                    curr = []
        if len(curr) > 0:
            phrases.append(curr)
        if debug:
            print(phrases)
        subphrases = []
        for i in range(len(phrases)):
            for j in range(1, len(phrases[i]) + 1):
                for k in range(len(phrases[i]) - j + 1):
                    sent = ' '.join(phrases[i][k:k + j])
                    subphrases.append((sent, (i, k, k + j), model.encode(sent)))
        similar_phrases = []
        for i in range(len(subphrases)):
            for j in range(i + 1, len(subphrases)):
                dist = 1 - distance.cosine(subphrases[i][2], subphrases[j][2])
                set1 = get_corefs(subphrases[i][0])
                set2 = get_corefs(subphrases[j][0])
                if dist > 0.7 and (((len(set1) == len(set2) == 0) and
                                    subphrases[j][1][0] > subphrases[i][1][0]) or subphrases[i][0] == subphrases[j][0]):
                    similar_phrases.append((dist, subphrases[i][0:2], subphrases[j][0:2]))
        similar_phrases.sort(key=lambda x: x[0], reverse=True)
        if debug:
            print(similar_phrases)
        edges = []
        nodes = []
        for _ in range(len(phrases)):
            nodes.append(Node())
        for phrase in similar_phrases:
            insert(phrase, edges, nodes)
        connected_components = []
        for edge in edges:
            component = []
            get_connected_component(edge, component, nodes)
            if len(component) > 1:
                connected_components.append(component)
        if debug:
            print("connected components:")
            print(connected_components)
        ans = ""
        phrase_size_count = 0
        phrase_count = 0
        # masked_content = ""
        next_token_dict = {}
        prev_token = ""
        prev_token_dict = {}
        for sent in ann.sentence:
            for token in sent.token:
                if token.lemma.lower() in sw_spacy or is_punctuation(token.word):
                    curr_token = token.word
                    if phrase_size_count > 0:
                        phrase_count += 1
                    phrase_size_count = 0
                else:
                    idx, is_start = get_component_index(phrase_count, phrase_size_count, connected_components)
                    if idx is None:
                        curr_token = token.word
                    elif is_start:
                        curr_token = "MSK<%d>" % idx
                        # if idx not in prev_token_dict:
                        #     prev_token_dict[idx] = set()
                        # prev_token_dict[idx].add(prev_token)
                    phrase_size_count += 1
                    # if idx is not None and nodes[phrase_count].marked_range[0] <= phrase_size_count < \
                    #         nodes[phrase_count].marked_range[1]:
                    #     masked_content += word.text
                    #     if nodes[phrase_count] == nodes[phrase_count].marked_range[1] - 1:
                    #         word_bank.append(masked_content)
                    #         masked_content = ""
                ans += curr_token
                if "MSK" in curr_token:
                    index = int(curr_token[4])
                    if index not in prev_token_dict:
                        prev_token_dict[index] = set()
                    prev_token_dict[index].add(prev_token)
                if "MSK" in prev_token:
                    index = int(prev_token[4])
                    if index not in next_token_dict:
                        next_token_dict[index] = set()
                    next_token_dict[index].add(curr_token)
                prev_token = curr_token
                ans += " "
        if "MSK" in prev_token:
            index = int(prev_token[4])
            if index not in next_token_dict:
                next_token_dict[index] = set()
            next_token_dict[index].add("")
        if debug:
            print("before removing subphrases: ", ans)
        ans = remove_repeated_subphrases(ans, next_token_dict, prev_token_dict)
        logger.warn("%s updated to %s", text, ans)
        return ans
    except:
        logger.info("got an error for string %s", text)
        return text


def update_csv_with_masked_content(path, article_col_name, model, client):
    df = pd.read_csv(path)
    masked_articles = [mask_out_content(article, model, client) for article in df[article_col_name]]
    df['masked_articles'] = masked_articles
    logger.info("completed conversion saving file to %s", path)
    df.to_csv(path)


def get_coref(text, ann, debug=False):
    chains = []
    for entry in ann.corefChain:
        chain = []
        for mention in entry.mention:
            chain.append((mention.sentenceIndex, (mention.beginIndex, mention.endIndex)))
        chains.append(chain)
    # doc = nlp(text)
    ans = ""
    # if debug:
    # print("printing doc sentences")
    # print("-----------------------")
    # # print([sent.text for sent in doc.sentences])

    for i, sent in enumerate(ann.sentence):
        if debug:
            print([(k, token.word) for k, token in enumerate(sent.token)])
        for j, token in enumerate(sent.token):
            idx, is_start = get_component_index(i, j, chains)
            print(i, j, idx, is_start)
            if idx is None:
                ans += token.word
            elif is_start:
                ans += "coref%d" % idx
            ans += " "
    # if debug:
    # print(ann.corefChain)
    # print("-----------------------")
    # print("printing ann sentences")
    # print("-----------------------")
    # for sentence in ann.sentence:
    #     print([(i, token.word) for i, token in enumerate(sentence.token)])
    # print("-----------------------")
    return ans


if __name__ == '__main__':
    # word_bank = []
    # nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', use_gpu=True)
    en = spacy.load('en_core_web_sm')
    sw_spacy = en.Defaults.stop_words
    parser = argparse.ArgumentParser()
    logger = get_logger(level='WARN')
    parser.add_argument("-p", "--path", help="path for input csv file")
    parser.add_argument("-a", "--article_col_name", help="column which contains the main text")
    parser.add_argument("-m", "--model", help="sentence transformer model")
    args = parser.parse_args()
    model = SentenceTransformer(args.model)
    client = CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'], timeout=30000)
    # update_csv_with_masked_content(args.path, args.article_col_name, model, client)
    # print(word_bank) pickle.dump(word_bank, open("../../data/word_bank.pkl", "wb"))
    text = "Every time I wash my car, it rains. Me washing my car has a definite effect on the weather."
    mask_out_content(text, model, client, debug=True)
