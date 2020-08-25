import pandas as pd
import csv


class SentenceGetter(object):

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
                sentence = sentence + word[0] + " "
            docs.append(sentence)
        return docs


def read_and_prepare_csv(filename, verbose=0):
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
