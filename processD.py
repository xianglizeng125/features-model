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
from keras.layers.recurrent import SimpleRNN
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras.preprocessing import sequence
import gensim
import logging


class process(object):
    def __init__(self): #because of self and will automatic operating in class process that point to __init__(self) content
        self.folderName=None
        stop_word = set(stopwords.words('english'))
        self.stop_word = [i.lower() for i in stop_word]
        self.lemmatizer = WordNetLemmatizer()
        self.positive=r'E://mydata//TinyData//pos' #C:\Users\xiangli125\Desktop\movie review\aclImdb\train\pos'
        self.negative=r'E://mydata//TinyData//neg' #C://Users//xiangli125//Desktop//TinyData//neg'
        self.doc_count=None
        self.tffreq={}
        self.tffreq_pos = {}
        self.tffreq_neg={}
        self.lookuptable=None
    def loadData(self,myfile):
        ff = open(myfile, 'r', encoding='utf-8')
        text = ff.read()
        ff.close()
        return text
    def clean(self, doc):
        tokens = self.tokenize(doc)
        valid= [i for i in tokens if i.lower() not in self.stop_word]
        valid=[i for i in valid if i.isalpha() and len(i)>2]
        lemma=[self.lemmatizer.lemmatize(word) for word in valid]

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
        mat=[self.lookuptable2[d] for d in document]
        return mat
    def matrix2(self, document):
        mat=[self.lookuptable[d] for d in document]
        return mat

    def document_level_pos_tags(self, document):

        return self.pos(document)
    def _document_level_pos_tags(self, document):
        #print(document)
        return self.pos_test(document)

    def _features(self):
        train_text_pos = self.All_training_review2("P")
        train_text_neg = self.All_training_review2("N")
        #y_train = array([1 for _ in range(90)] + [0 for _ in range(90)])
        joinText_pos = list(itertools.chain(*train_text_pos))
        self.tffreq_pos = self.frequency(' '.join(joinText_pos))
        joinText_neg = list(itertools.chain(*train_text_neg))
        self.tffreq_neg = self.frequency(' '.join(joinText_neg))
        _size=len(joinText_pos)+len(joinText_neg)
        uniquewords_pos = list(set(joinText_pos))
        uniquewords_neg = list(set(joinText_neg))
        uniquewords = list(set(uniquewords_pos + uniquewords_neg))
        model = gensim.models.Word2Vec([uniquewords], sg=1, min_count=1, window=5, hs=1, size=100, workers=2,
                                       negative=0)
        idf_pos = {}
        idf_neg = {}
        for i in uniquewords_pos:
            idf_pos[i] = 0
        for i,j in zip(uniquewords_pos, train_text_pos):
            if i in j:
                idf_pos[i] += 1
        for i in uniquewords_neg:
            idf_neg[i] = 0
        for i,j in zip(uniquewords_neg, train_text_neg):
            if i in j:
                idf_neg[i] += 1
        tfidf_dict_pos = {}
        tfidf_dict_neg = {}
        tfidfscore_pos = [(word, self.idf(word, idf_pos) * self.tf(word, self.tffreq_pos)) for word in uniquewords_pos]
        tfidfscore_neg = [(word, self.idf(word, idf_neg) * self.tf(word, self.tffreq_neg)) for word in uniquewords_neg]
        for x in tfidfscore_pos:
            tfidf_dict_pos[x[0]] = x[1]
        for x in tfidfscore_neg:
            tfidf_dict_neg[x[0]] = x[1]

        pos_document = {}
        neg_document = {}
        _pos=[]
        _neg=[]
        cnt_pos=0
        cnt_neg = 0

        for i in train_text_pos:
            pos_dict_pos = self.pos(i)
            k = pos_dict_pos.keys()
            v = pos_dict_pos.values()
            temp = [model[j].tolist()for j in k]
            temp2 = [tfidf_dict_pos[j]for j in k]

            _pos.append(list(v) + list(itertools.chain(*temp))+temp2)
        for i in _pos:
            pos_document[cnt_pos]=(i, 1)
            cnt_pos+=1
       # print(len(pos_document))
        for i in train_text_neg:
            pos_dict_neg = self.pos(i)
            k = pos_dict_neg.keys()
            v = pos_dict_neg.values()
            temp = [model[j].tolist() for j in k]
            temp2 = [tfidf_dict_neg[j] for j in k]
            for j in k:
                temp.append(model[j].tolist())
            _neg.append(list(v) + list(itertools.chain(*temp))+temp2)
        for i in _neg:
            neg_document[cnt_neg]=(i, 0)
            cnt_neg+=1

        xtrain=[i[0] for i in pos_document.values()]

        ytrain=[i[1] for i in pos_document.values()]

        for i in neg_document.values():
            xtrain.append(i[0])
            ytrain.append(i[1])

        return xtrain, ytrain, _size

    def _features_test(self):
        test_text_pos = self.All_testing_review("P")
        test_text_neg = self.All_testing_review("N")

        A_test_text = test_text_pos+test_text_neg

        joinText_pos = list(itertools.chain(*test_text_pos))
        self.tffreq_pos = self.frequency(' '.join(joinText_pos))
        joinText_neg = list(itertools.chain(*test_text_neg))
        self.tffreq_neg = self.frequency(' '.join(joinText_neg))
        _size = len(joinText_pos) + len(joinText_neg)

        uniquewords_pos = list(set(joinText_pos))
        uniquewords_neg = list(set(joinText_neg))
        uniquewords = list(set(uniquewords_pos + uniquewords_neg))
        model = gensim.models.Word2Vec([uniquewords], sg=1, min_count=1, window=5, hs=1, size=100, workers=2,
                                       negative=0)
        idf_pos = {}
        idf_neg = {}
        for i in uniquewords_pos:
            idf_pos[i] = 0
        for i, j in zip(uniquewords_pos, test_text_pos):
            if i in j:
                idf_pos[i] += 1
        for i in uniquewords_neg:
            idf_neg[i] = 0
        for i, j in zip(uniquewords_neg, test_text_neg):
            if i in j:
                idf_neg[i] += 1
        tfidf_dict_pos = {}
        tfidf_dict_neg = {}
        tfidfscore_pos = [(word, self.idf(word, idf_pos) * self.tf(word, self.tffreq_pos)) for word in uniquewords_pos]
        tfidfscore_neg = [(word, self.idf(word, idf_neg) * self.tf(word, self.tffreq_neg)) for word in uniquewords_neg]
        for x in tfidfscore_pos:
            tfidf_dict_pos[x[0]] = x[1]
        for x in tfidfscore_neg:
            tfidf_dict_neg[x[0]] = x[1]

        pos_document = {}
        neg_document = {}
        _pos = []
        _neg = []
        cnt_pos = 0
        cnt_neg = 0

        for i in test_text_pos:
            pos_dict_pos = self.pos_test(i)
            k = pos_dict_pos.keys()
            v = pos_dict_pos.values()
            temp = [model[j].tolist() for j in k]
            temp2 = [tfidf_dict_pos[j] for j in k]

            _pos.append(list(v) + list(itertools.chain(*temp)) + temp2)
        for i in _pos:
            pos_document[cnt_pos] = (i, 1)
            cnt_pos += 1
        # print(len(pos_document))
        for i in test_text_neg:
            pos_dict_neg = self.pos_test(i)
            k = pos_dict_neg.keys()
            v = pos_dict_neg.values()
            temp = [model[j].tolist() for j in k]
            temp2 = [tfidf_dict_neg[j] for j in k]
            for j in k:
                temp.append(model[j].tolist())
            _neg.append(list(v) + list(itertools.chain(*temp)) + temp2)
        for i in _neg:
            neg_document[cnt_neg] = (i, 0)
            cnt_neg += 1

        xtest = [i[0] for i in pos_document.values()]

        ytest = [i[1] for i in pos_document.values()]

        for i in neg_document.values():
            xtest.append(i[0])
            ytest.append(i[1])

        return xtest, ytest, _size, A_test_text

    def pos(self, text):
        get_all_text = text #self.All_training_review()
        Ptagged = []
        for x in range(len(get_all_text)):
            Ptagged.append(nltk.pos_tag(nltk.word_tokenize(str(get_all_text[x]))))
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
            Ptagged.append(nltk.pos_tag(nltk.word_tokenize(str(get_all_text[x]))))
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

    def All_training_review2(self, tag):
        if tag=="P":
            positive_doc = self.operator(self.positive, True)
            return positive_doc
        else:
            negative_doc = self.operator(self.negative, True)
            return  negative_doc

    def All_testing_review(self, tag):
        if tag=="P":
            positive_test = self.operator(self.positive, False)
            return positive_test
        else:
            negative_test = self.operator(self.negative, False)
            return  negative_test

    def model_t(self):
        x_train = sequence.pad_sequences(self._features()[0], maxlen=408, dtype='float32')

        x_test = sequence.pad_sequences(self._features_test()[0], maxlen=408, dtype='float32')

        _model = Sequential()
        _model.add(Embedding(output_dim=32, input_dim=self._features()[2], input_length=408))
        _model.add(Dropout(0.2))

        _model.add(LSTM(32))

       # _model.add(Flatten())
        _model.add(Dense(units=256, input_dim=784,kernel_initializer='normal',activation='relu'))
        _model.add(Dropout(0.35))

        _model.add(Dense(units=1, kernel_initializer='normal',activation='sigmoid'))

        _model.summary()

        _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_history = _model.fit(x_train, self._features()[1], batch_size=10
                                   , epochs=10, verbose=2)
        scores = _model.evaluate(x_test, self._features_test()[1], verbose=1)
        print(scores)
        predict = _model.predict_classes(x_test) #2 dimensi array
        print(predict[:10])

        predict_classes = predict.reshape(-1) #untuk mengubah 1 dimensi array
        print(predict_classes[:10])

        sentimentDict = {1:"positive",0:"negative"}
        def display_test(i):
            print(self._features_test()[3][i])   #feature_test[3] = A_text_test
            print("Actual Label Result = ", sentimentDict[self._features_test()[1][i]],"\nPrediction Result = ", sentimentDict[predict_classes[i]])
        display_test(1)


        '''
        model_json = _model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        _model.save_weights("model.h5")

        json_f = open("model.json", "r")
        load_model_json = json_file.read()
        json_file.close()
        load_md = model_from_json(load_model_json)
        '''


obj = process()
print(obj.model_t())




