#! /usr/bin/env python3
import keras
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
import  keras.backend as K

import numpy as np

import gzip, pickle


class Text2Dataset:
    def __init__(self, wordNgrams=1, label_prefix='__label__', minCount=1):
        self.wordNgrams = wordNgrams
        self.label_prefix = label_prefix
        self.minCount = minCount

        self.word2idx = None
        self.words2idx = None
        self.label2idx = None
        self.idx2label = None
        self.train_X = None
        self.train_y = None
        self.max_features = None
        self.token_indice = None

    def create_ngram_set(self, input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, self.wordNgrams + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in self.token_indice:
                        new_list.append(self.token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def text2List(self, text_path):
        with open(text_path) as f:
            lines = f.readlines()

        getLabel = lambda line: [words.strip() for words in line.split(',') if self.label_prefix in words][0]
        getWords = lambda line: (','.join([words for words in line.split(',') if self.label_prefix not in words])
                                 .strip().replace('\n', ''))

        label_list = [getLabel(line) for line in lines]
        words_list = [getWords(line) for line in lines]

        return words_list, label_list

    def loadTrain(self, text_path):
        words_list, label_list = self.text2List(text_path)

        label_words_dict = {}
        for label, words in zip(label_list, words_list):
            if len(words) > self.minCount:
                if label in label_words_dict:
                    label_words_dict[label].append(words)
                else:
                    label_words_dict[label] = []

        self.label2idx = {label: idx for idx, label in enumerate(label_words_dict)}
        self.idx2label = {self.label2idx[label]: label for label in self.label2idx}
        self.word2idx = {word: idx for idx, word in enumerate(set(' '.join(words_list).split()))}
        self.words2idx = lambda words: [self.word2idx[word] for word in words.split() if word in self.word2idx]

        self.train_X = [self.words2idx(words) for words in words_list]
        self.train_y = [self.label2idx[label] for label in label_list]
        self.max_features = len(self.word2idx)

        if self.wordNgrams > 1:
            print('Adding {}-gram features'.format(self.wordNgrams))
            ngram_set = set()
            for input_list in self.train_X:
                for i in range(2, self.wordNgrams + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            start_index = self.max_features + 1
            self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {self.token_indice[k]: k for k in self.token_indice}

            self.max_features = np.max(list(indice_token.keys())) + 1

            self.train_X = self.add_ngram(self.train_X)

        return self.train_X, self.train_y

    def loadTest(self, text_path):
        words_list, label_list = self.text2List(text_path)

        test_X = [self.words2idx(words) for words in words_list]
        test_y = [self.label2idx[label] for label in label_list]
        test_X = self.add_ngram(test_X)

        return test_X, test_y

class FastText:
    def __init__(self, wordNgrams=1, label_prefix='__label__', minCount=1, args=None):
        if args is None:
            self.text2Dataset = Text2Dataset(wordNgrams, label_prefix, minCount)

            self.max_features = None
            self.maxlen = None
            self.batch_size = None
            self.embedding_dims = None
            self.epochs = None
            self.lr = None
            self.num_classes = None
            self.model = None
        else:
            (wordNgrams, label_prefix, minCount, word2idx, label2idx, token_indice,
             self.max_features, self.maxlen, self.batch_size, self.embedding_dims,
             self.epochs, self.lr, self.num_classes, model_weights) = args

            self.text2Dataset = Text2Dataset(wordNgrams, label_prefix, minCount)
            self.text2Dataset.words2idx = lambda words: [word2idx[word] for word in words.split() if word in word2idx]
            self.text2Dataset.label2idx = label2idx
            self.text2Dataset.idx2label = {label2idx[label]: label for label in label2idx}
            self.text2Dataset.token_indice = token_indice
            self.model = self.build_model(model_weights)

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def build_model(self, weights=None):
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=self.lr),
                metrics=[self.precision, self.recall])

        if weights is not None:
            model.set_weights(weights)
        return model

    def train(self, text_path, maxlen=400, batch_size=32, embedding_dims=100, epochs=5, lr=0.001, verbose=1):
        train_X, train_y = self.text2Dataset.loadTrain(text_path)

        self.max_features = self.text2Dataset.max_features
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.lr = lr
        self.num_classes = len(set(train_y))

        self.model = self.build_model()

        train_X = sequence.pad_sequences(train_X, maxlen=self.maxlen)
        train_Y = to_categorical(train_y, self.num_classes)

        self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        return self

    def test(self, text_path, verbose=1):
        test_X, test_y = self.text2Dataset.loadTest(text_path)
        test_X = sequence.pad_sequences(test_X, maxlen=self.maxlen)
        test_Y = to_categorical(test_y, self.num_classes)
        
        c, p, r = self.model.evaluate(test_X, test_Y, batch_size=self.batch_size, verbose=verbose)

        print("N\t" + str(len(test_y)))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))
        return str(len(test_y)), p, r

    def save_model(self, path):
        args = (self.text2Dataset.wordNgrams, self.text2Dataset.label_prefix,
                self.text2Dataset.minCount, self.text2Dataset.word2idx,
                self.text2Dataset.label2idx, self.text2Dataset.token_indice,
                self.max_features, self.maxlen, self.batch_size, self.embedding_dims,
                self.epochs, self.lr, self.num_classes, self.model.get_weights())
        with gzip.open(path, 'wb') as f:
            pickle.dump(args, f)

    def predict(self, text, k=1):
        text = ','.join([words for words in text.split(',')]).strip().replace('\n', '')
        X = self.text2Dataset.words2idx(text)
        X = self.text2Dataset.add_ngram([X])
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        predict = self.model.predict(X).flatten()
        results = [(self.text2Dataset.idx2label[idx], predict[idx]) for idx in range(len(predict))]
        return sorted(results, key=lambda item: item[1], reverse=True)[:k]

def train_supervised(input, lr=0.01, dim=100, epoch=5, minCount=1, wordNgrams=1, label='__label__', verbose=1, maxlen=400):
    fastText = FastText(wordNgrams=wordNgrams, label_prefix=label, minCount=minCount)
    fastText.train(text_path=input, maxlen=maxlen, embedding_dims=dim, epochs=epoch, lr=lr, verbose=verbose)
    return fastText

def load_model(path):
    with gzip.open(path, 'rb') as f:
        args = pickle.load(f)

    fastText = FastText(args=args)
    return fastText


if __name__ == '__main__':
    data_path = './classifier_data.txt'
    model_path = 'tmp/FastText.bin.gz'
    text = '''birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta high school in 
    lakewood township new jersey . it was founded by rabbi shmuel zalmen stein in 2001 after his 
    father rabbi chaim stein asked him to open a branch of telshe yeshiva in lakewood . 
    as of the 2009-10 school year the school had an enrollment of 76 students and 6 . 6 classroom 
    teachers ( on a fte basis ) for a student–teacher ratio of 11 . 5 1 .'''

    model = train_supervised(data_path, wordNgrams=2, lr=0.01, epoch=50, minCount=5)
    model.save_model(model_path)
    model2 = load_model(model_path)
    model2.test(data_path)

    print('Predict:', model2.predict(text, k=3))
