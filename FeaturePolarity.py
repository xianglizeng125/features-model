from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os, nltk
import itertools
import math
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from numpy import array
import gensim
import logging
from collections import defaultdict

class process(object):
    def __init__(self): #self  will automatic operating in class process that point to __init__(self) content
        self.folderName=None
        stop_word = set(stopwords.words('english'))
        self.stop_word = [i.lower() for i in stop_word]
        self.lemmatizer = WordNetLemmatizer()
        self.positive=r'E://mydata//TinyData//pos'
        self.negative=r'E://mydata//TinyData//neg' #C://Users//xiangli125//Desktop//TinyData//neg'
        self.doc_count=None
        self.tffreq={}
        self.lookuptable=None
    def loadData(self,myfile):
        ff = open(myfile, 'r', encoding='utf-8')
        text = ff.read()
        ff.close()
        return text
    def clean(self, doc):
        #remove unwanted tokens
        tokens = self.tokenize(doc)
        valid= [i for i in tokens if i.lower() not in self.stop_word]
        #translation = str.maketrans("", "", string.punctuation)
        #valid = [w.translate(translation) for w in valid]
        valid=[i for i in valid if i.isalpha() and len(i)>2]
        lemma=[self.lemmatizer.lemmatize(word) for word in valid]
        #print(lemma)
        return lemma
    def operator(self, directory, is_trian):
        document = []
        for myfile in os.listdir(directory):
            if is_trian and myfile.startswith("CV"):
                continue
            if not is_trian and not myfile.startswith("CV"):
                continue
            path = directory + "//" + myfile  # create a full path  pada file untuk di buka
            doc = self.loadData(path)
            tokenz = self.clean(doc)
            document.append(tokenz)
            self.doc_count=document
        return document
    def tokenize(self, doc):
        lowertext = doc.lower()
        tokens= word_tokenize(lowertext)
        return tokens
    def frequency(self, text):
        _text= self.clean(text)
        count = Counter(_text)
        return count

    def tf(self,word, count):
        return count[word]/ sum(count.values())
    def n_containing(self,word, count_list):
        return sum(1 for count in count_list if word in count)
    def idf(self,word, count_list):
        return math.log(len(count_list))/(1+count_list[word])
    def tfidf(self,word, count):#, count_list):
        return self.idf(word, count)*self.tf(word,count)

    def matrix(self, document):
        mat=[self.lookuptable[d] for d in document]
        return mat
    def matrix_test(self, document):
        mat=[self.lookuptable[d] for d in document]
        return mat

    def features(self):
        train_text = self.All_training_review()
        joinText =list(itertools.chain(*train_text))
        _vocabsize = len(joinText)
        self.tffreq=self.frequency(' '.join(joinText))    #dapat term frequency each word  type = class  collectioncounter
        uniquewords = list(set(joinText))
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = gensim.models.Word2Vec([uniquewords], sg=1, min_count=1,window=5,hs=1, size=100,workers=2, negative=0)
        w2v_dim_dict = {}
        tfidf_dict = {}
        idf = {}
        get_ulf = defaultdict(list)#{}
        pos_dict = self.pos_train(uniquewords)
        for i in uniquewords:
            idf[i] = 0
        for i,j in zip(uniquewords, train_text):
            if i in j:
                idf[i] += 1
        for x in uniquewords:
            w2v_dim_dict[x] = model[x]
        tfidfscore = [(word, self.idf(word, idf) * self.tf(word, self.tffreq)) for word in uniquewords]
        for x in tfidfscore:
            tfidf_dict[x[0]] = x[1]
        for item in (tfidf_dict,pos_dict,w2v_dim_dict):
            for k,v in item.items():
                get_ulf[k].append(v)
        self.lookuptable = get_ulf
        _all=[self.matrix(i)for i in train_text]
        _get_l = []
        for nlist in _all:
            temp = []
            for item in nlist:
                temp.append(item[0])
                temp.append(item[1])
                temp.extend(item[2].tolist())
            _get_l.append(temp)

        return _get_l, _vocabsize

    def features_test(self):
        test_text = self.All_testing_review()
        joinText = list(itertools.chain(*test_text))
        _vocabsize_test = len(joinText)
        self.tffreq = self.frequency(
            ' '.join(joinText))  # dapat term frequency each word  type = class  collectioncounter
        uniquewords = list(set(joinText))
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = gensim.models.Word2Vec([uniquewords], sg=1, min_count=1, window=5, hs=1, size=100, workers=2,
                                       negative=0)
        w2v_dim_dict = {}
        tfidf_dict = {}
        idf = {}
        get_ulf = defaultdict(list)  # {}
        pos_dict = self.pos_test(uniquewords)
        for i in uniquewords:
            idf[i] = 0
        for i, j in zip(uniquewords, test_text):
            if i in j:
                idf[i] += 1
        for x in uniquewords:
            w2v_dim_dict[x] = model[x]
        tfidfscore = [(word, self.idf(word, idf) * self.tf(word, self.tffreq)) for word in uniquewords]
        for x in tfidfscore:
            tfidf_dict[x[0]] = x[1]
        for item in (tfidf_dict, pos_dict, w2v_dim_dict):
            for k, v in item.items():
                get_ulf[k].append(v)
        self.lookuptable = get_ulf
        _all = [self.matrix_test(i) for i in test_text]
        _get_feature_test = []
        for nlist in _all:
            temp = []
            for item in nlist:
                temp.append(item[0])
                temp.append(item[1])
                temp.extend(item[2].tolist())
            _get_feature_test.append(temp)

        return _get_feature_test, _vocabsize_test

    def model_t(self):

        x_train = sequence.pad_sequences(self.features()[0], maxlen=408, dtype='float32') #x_train type is numpy array
        y_train = array([1 for _ in range(90)] + [0 for _ in range(90)])

        x_test = sequence.pad_sequences(self.features_test()[0],maxlen=408, dtype='float32')
        y_test = array([1 for _ in range(10)] + [0 for _ in range(10)])

        _model = Sequential()
        _model.add(Embedding(output_dim=32, input_dim=self.features()[1], input_length=408))
        _model.add(Dropout(0.2))

        _model.add(Flatten())
        _model.add(Dense(units=256, activation='relu'))
        _model.add(Dropout(0.35))

        _model.add(Dense(units=1, activation='sigmoid'))
        _model.summary()

        _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_history = _model.fit(x_train, y_train, batch_size=10, epochs=50, verbose=2)
        print(train_history)
        scores = _model.evaluate(x_test,y_test,verbose=1)
        print(scores)
        return train_history

    def pos_train(self, text):
        get_all_text = text #self.All_training_review()
        Ptagged = []
        for x in range(len(get_all_text)):
            Ptagged.append(nltk.pos_tag(nltk.word_tokenize(get_all_text[x])))
        d = dict([(y, x + 1) for x, y in enumerate(sorted(set(list(itertools.chain(*Ptagged)))))])
        mydict = {}
        for i, j in d.items():
            # print(i[1])
            if i[1] in 'JJ':
                mydict[i[0]] = 1
            elif i[1] in 'VBG':
                mydict[i[0]] = 2
            elif i[1] in 'VBN':
                mydict[i[0]] = 3
            elif i[1] in 'VBD':
                mydict[i[0]] = 4
            elif i[1] in 'VBP':
                mydict[i[0]] = 5
            elif i[1] in 'VBZ':
                mydict[i[0]] = 6
            elif i[1] in 'NN':
                mydict[i[0]] = 7
            elif i[1] in 'NNP':
                mydict[i[0]] = 8
            elif i[1] in 'JJS':
                mydict[i[0]] = 9
            else:
                mydict[i[0]] = 0

        factor = 1.0/sum(mydict.values())
        for k in mydict:
            mydict[k]=mydict[k]*factor
        return mydict
    def pos_test(self, text):
        get_all_text = text  # self.All_training_review()
        Ptagged = []
        for x in range(len(get_all_text)):
            Ptagged.append(nltk.pos_tag(nltk.word_tokenize(get_all_text[x])))
        d = dict([(y, x + 1) for x, y in enumerate(sorted(set(list(itertools.chain(*Ptagged)))))])
        mydict = {}
        for i, j in d.items():
            # print(i[1])
            if i[1] in 'JJ':
                mydict[i[0]] = 1
            elif i[1] in 'VBG':
                mydict[i[0]] = 2
            elif i[1] in 'VBN':
                mydict[i[0]] = 3
            elif i[1] in 'VBD':
                mydict[i[0]] = 4
            elif i[1] in 'VBP':
                mydict[i[0]] = 5
            elif i[1] in 'VBZ':
                mydict[i[0]] = 6
            elif i[1] in 'NN':
                mydict[i[0]] = 7
            elif i[1] in 'NNP':
                mydict[i[0]] = 8
            elif i[1] in 'JJS':
                mydict[i[0]] = 9
            else:
                mydict[i[0]] = 0

        factor = 1.0 / sum(mydict.values())
        for k in mydict:
            mydict[k] = mydict[k] * factor
        return mydict
    def All_training_review(self):
        positive_doc = self.operator(self.positive, True)
        negative_doc = self.operator(self.negative, True)
        train_doc = positive_doc + negative_doc
        return train_doc
    def All_testing_review(self):
        positive_doc = self.operator(self.positive, False)
        negative_doc = self.operator(self.negative, False)
        test_doc = positive_doc+negative_doc
        return test_doc

obj = process()

#print(obj.features_test())
print(obj.model_t())



