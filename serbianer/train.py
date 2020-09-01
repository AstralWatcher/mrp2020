from serbianer.load_data.load_dataset import read_and_prepare_csv, SentenceGetter, bert_load_index_tags
from serbianer.vocab import Vocab
import matplotlib.pyplot as plt
import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig

configuration = BertConfig()  # default parameters and configuration for BERT


def create_model(model='bert-base-multilingual-cased'):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(model)

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


if __name__ == "__main__":
    debug = False
    embedding_type = "Bert"
    saving_index_path = '../datasets/vocab/bert_idx2tag.csv'
    max_len = 75
    data = read_and_prepare_csv("../datasets/hr500k.csv")
    vocab_dicts = Vocab(data, saving_index_path)
    if debug:
        vocab_dicts.display_hist()

    docs = vocab_dicts.get_docs()

    save_path = "../bert_model/"
    model_name = 'bert-base-multilingual-cased'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        if not os.path.exists(save_path + model_name):
            slow_tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer(save_path + "/vocab.txt", lowercase=False)
