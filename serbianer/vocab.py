from serbianer.load_data.load_dataset import SentenceGetter, bert_load_index_tags
import matplotlib.pyplot as plt
import os


class Vocab:

    def __init__(self, data, saving_index_path, save=True, force_compute=False, preload=True):
        """
        :param data, pandas dataframe
        :param saving_index_path, path to where idx2tag and word2idx will be
        :param save, if true will use saving_index_path to save idx2tag and word2idx
        :param force_compute, make from data new id2tag and word2idx
        :param preload: call prepare_vocab()
        """
        self.data = data
        self.n_words = 0
        self.n_tags = 0
        self.sentences = None
        self.saving_index_path = saving_index_path
        self.saving = save
        self.force_compute = force_compute
        self.docs = None
        if preload:
            self._prepare_vocab()

    def get_n_words(self):
        return self.n_words

    def get_n_tags(self):
        return self.n_tags

    def get_docs(self):
        return self.docs

    def get_sentences(self):
        return self.sentences

    def display_hist(self, bins=45):
        len_sens = [len(sen) for sen in self.sentences]
        largest_sen = max(len_sens)
        plt.xlabel('Number of words in sentences')
        plt.ylabel('Number of sentences that have that much words')
        plt.title('Largest sentence has ' + str(largest_sen))
        plt.hist(len_sens, bins=bins)
        plt.show()

    def _prepare_vocab(self):
        # self.data.tail(10)

        words = list(set(self.data["Word"].values))
        self.n_words = len(words)

        tags = list(set(self.data["Tag"].values))
        self.n_tags = len(tags)

        getter = SentenceGetter(self.data)
        self.sentences = getter.sentences  # getter.get_next() get one
        self.docs = getter.get_docs()
        word2idx = {w: i + 2 for i, w in enumerate(words)}
        word2idx["PAD"] = 0
        word2idx["UNK"] = 1
        idx2word = {i: w for w, i in word2idx.items()}

        if os.path.exists(path=self.saving_index_path) and not self.force_compute:
            idx2tag, tag2idx = bert_load_index_tags(self.saving_index_path)
            saving = False
        else:
            tag2idx = {t: i + 1 for i, t in enumerate(tags)}
            tag2idx["PAD"] = 0
            idx2tag = {i: w for w, i in tag2idx.items()}
            saving = True

        if saving and self.saving:
            with open(self.saving_index_path, 'w') as f:
                writing = [str(key) + "," + str(value) + "\n" for key, value in idx2tag.items()]
                f.writelines(writing)

        return word2idx, idx2tag, idx2word
