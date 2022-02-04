import argparse
from logicedu import get_logger
import stanza
import spacy
import string
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import pandas as pd
import pickle


class Node:
    def __init__(self):
        self.visited = False
        self.edges = []
        self.marked_range = None


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


def visit(i, range, component, nodes):
    if nodes[i].visited:
        return
    nodes[i].visited = True
    nodes[i].marked_range = range
    component.append(i)
    for edge in nodes[i].edges:
        visit(edge.output_index, edge.output_range, component, nodes)


def get_connected_component(edge, component, nodes):
    visit(edge.input_index, edge.input_range, component, nodes)


def get_component_index(phrase_count, connected_components):
    for i in range(len(connected_components)):
        if phrase_count in connected_components[i]:
            return i
    return None


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words


def mask_out_content(text, model):
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
                if phrase_size_count == 0:
                    idx = get_component_index(phrase_count, connected_components)
                if idx is None or phrase_size_count < nodes[phrase_count].marked_range[0] or phrase_size_count >= \
                        nodes[phrase_count].marked_range[
                            1]:
                    ans += word.text
                elif phrase_size_count == nodes[phrase_count].marked_range[0]:
                    ans += "MSK<%d>" % idx
                if idx is not None and nodes[phrase_count].marked_range[0] <= phrase_size_count < \
                        nodes[phrase_count].marked_range[1]:
                    masked_content += word.text
                    if nodes[phrase_count] == nodes[phrase_count].marked_range[1] - 1:
                        word_bank.append(masked_content)
                        masked_content = ""
                phrase_size_count += 1
            ans += " "
    logger.warn("%s updated to %s", text, ans)
    return ans


def update_csv_with_masked_content(path, article_col_name, model):
    df = pd.read_csv(path)
    masked_articles = [mask_out_content(article, model) for article in df[article_col_name][0:100]]
    # df['masked_articles'] = masked_articles
    # df.to_csv(path)


if __name__ == '__main__':
    word_bank = []
    parser = argparse.ArgumentParser()
    logger = get_logger(level='WARN')
    parser.add_argument("-p", "--path", help="path for input csv file")
    parser.add_argument("-a", "--article_col_name", help="column which contains the main text")
    parser.add_argument("-m", "--model", help="sentence transformer model")
    args = parser.parse_args()
    model = SentenceTransformer(args.model)
    update_csv_with_masked_content(args.path, args.article_col_name, model)
    print(word_bank)
    pickle.dump(word_bank, open("../../data/word_bank.pkl", "wb"))
