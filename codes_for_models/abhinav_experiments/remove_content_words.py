import argparse
from logicedu import get_logger, add
import stanza
import spacy
import string
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import pandas as pd
import pickle
from stanza.server import CoreNLPClient


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
        if (range[1] > new_range[0] >= range[0]) or (range[0] < new_range[1] <= range[1]):
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


def mask_out_content(text, model, client):
    text = get_coref(text, client)
    doc = nlp(text)
    phrases = []
    curr = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma.lower() not in sw_spacy and not is_punctuation(word.text):
                curr.append(word.lemma.lower())
            elif len(curr) > 0:
                phrases.append(curr)
                curr = []
    if len(curr) > 0:
        phrases.append(curr)
    # print(phrases)
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
            if dist > 0.7 and subphrases[j][1][0] > subphrases[i][1][0]:
                similar_phrases.append((dist, subphrases[i][0:2], subphrases[j][0:2]))
    similar_phrases.sort(key=lambda x: x[0], reverse=True)
    # print(similar_phrases)
    edges = []
    nodes = []
    for i in range(len(phrases)):
        nodes.append(Node())
    for phrase in similar_phrases:
        insert(phrase, edges, nodes)
    connected_components = []
    for edge in edges:
        component = []
        get_connected_component(edge, component, nodes)
        if len(component) > 1:
            connected_components.append(component)
    # print(connected_components)
    ans = ""
    phrase_size_count = 0
    phrase_count = 0
    idx = None
    masked_content = ""
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma.lower() in sw_spacy or is_punctuation(word.text):
                ans += word.text
                if phrase_size_count > 0:
                    phrase_count += 1
                phrase_size_count = 0
            else:
                idx, is_start = get_component_index(phrase_count, phrase_size_count, connected_components)
                if idx is None:
                    ans += word.text
                elif is_start:
                    ans += "MSK<%d>" % idx
                # if idx is not None and nodes[phrase_count].marked_range[0] <= phrase_size_count < \
                #         nodes[phrase_count].marked_range[1]:
                #     masked_content += word.text
                #     if nodes[phrase_count] == nodes[phrase_count].marked_range[1] - 1:
                #         word_bank.append(masked_content)
                #         masked_content = ""
                phrase_size_count += 1
            ans += " "
    logger.warn("%s updated to %s", text, ans)
    return ans


def update_csv_with_masked_content(path, article_col_name, model, client):
    df = pd.read_csv(path)
    masked_articles = [mask_out_content(article, model, client) for article in df[article_col_name]]
    df['masked_articles'] = masked_articles
    df.to_csv(path)


def get_coref(text, client):
    ann = client.annotate(text)
    chains = []
    for entry in ann.corefChain:
        chain = []
        for mention in entry.mention:
            chain.append((mention.sentenceIndex, (mention.beginIndex, mention.endIndex)))
        chains.append(chain)
    doc = nlp(text)
    ans = ""
    for i, sent in enumerate(doc.sentences):
        for j, word in enumerate(sent.words):
            idx, is_start = get_component_index(i, j, chains)
            if idx is None:
                ans += word.text
            elif is_start:
                ans += "coref%d" % idx
            ans += " "
    return ans


if __name__ == '__main__':
    # word_bank = []
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', use_gpu=True)
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
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'])
    update_csv_with_masked_content(args.path, args.article_col_name, model, client)
    # print(word_bank)
    # pickle.dump(word_bank, open("../../data/word_bank.pkl", "wb"))
