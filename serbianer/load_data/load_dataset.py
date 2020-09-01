import pandas as pd
import csv


class SentenceGetter(object):
    """
    Group objects in dataframe to sentences.
     e.g. 1 so we get sentences 0 with for example 74 words
    """

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["Pos"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def get_docs(self):
        docs = []
        for s in self.sentences:
            sentence = ""
            for word in s:
                if word[1] == 'Z':
                    if word[0] in ('[', '(', '"'):
                        sentence = sentence.rstrip() + word[0]
                    else:
                        sentence = sentence.rstrip() + word[0] + " "
                else:
                    sentence = sentence + word[0] + " "

            docs.append(sentence.rstrip())
        return docs


def read_and_prepare_csv(filename, verbose=0):
    """
    Read csv file and make Pandas Dataframe
    :param filename: path to a csv file for ner
    :param verbose: will it loaded sentences
    :return: Pandas Dataframe with Sentence,Word,Tag structure
    """
    df = pd.read_csv(filename, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)
    if verbose:
        print(df.head())
        print(df.isnull().sum())
    df = df.fillna(method='ffill')  # fills the NaN's
    x, y, z = df['Sentence #'].nunique(), df.Word.nunique(), df.Tag.nunique()
    if verbose:
        print("{0} Sentences, {1} Words, {2} Tags".format(x, y, z))
    if verbose:
        print(df.groupby('Tag').size().reset_index(name='counts'))  # distribution of tags

    return df


def bert_load_index_tags(path='../datasets/vocab/bert_idx2tag.csv'):
    """
    Return saved index and tags embedded for words
    :param path: path to saved
    :return: Index2tag dict and Tag2index dict
    """
    index2tag = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            to_file = lines[i].split(',')
            index2tag[int(to_file[0])] = to_file[1].replace("\n", "")
    tag2idx = {value: key for key, value in index2tag.items()}
    return index2tag, tag2idx
