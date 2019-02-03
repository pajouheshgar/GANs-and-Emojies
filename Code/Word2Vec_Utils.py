import re
import json
import codecs
import numpy as np
from tqdm import *


def load_word2vec(word2vec_file_address, word_dict):
    primary_word2embedding = {}
    secondary_word2embedding = {}
    print("Preparing word2vec")
    with codecs.open(word2vec_file_address, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        counter = 0
        for line, time in zip(range(vocab_size), trange(vocab_size)):
            counter += 1
            word = []
            while True:
                ch = chr(ord(f.read(1)))
                if ch == ' ':
                    word = "".join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word.lower() in word_dict:
                pass
            else:
                f.read(binary_len)
                continue

            embedding = np.fromstring(f.read(binary_len), dtype='float32')
            if word == word.lower():
                primary_word2embedding[word] = embedding
            else:
                secondary_word2embedding[word.lower()] = embedding

    index = 0
    for w in secondary_word2embedding.keys():
        if w not in primary_word2embedding:
            primary_word2embedding[w] = secondary_word2embedding[w]
            index += 1

    return primary_word2embedding


if __name__ == "__main__":
    word2vec_file = '../../Datasets/Word2Vec/GoogleNews-vectors-negative300.bin'
    word_dict = {}

    with open("../Dataset/emoji_pretty.json") as f:
        additional_data = json.load(f)
        img_name_2_info = {d['image']: d for d in additional_data}


    def get_words_list(info):
        name = info['name']
        if name is None:
            name = info['short_name']
        words_list = re.split("-|_| ", name)
        words_list = [w.lower() for w in words_list]
        return words_list


    for k in img_name_2_info.keys():
        for w in get_words_list(img_name_2_info[k]):
            word_dict[w] = 1

    word_embedding = load_word2vec(word2vec_file, word_dict)
