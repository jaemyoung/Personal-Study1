
#pip install gensim==3.4.0
#pip install smart_open==1.9.0
"""
class Doc_to_vec :
    def __init__(self, document, epochs=100, embedding_size=64, alpha=0.01, min_alpha=0.00025, min_count=1, dm=1, model_name=None) :
        # data
        self.document = document
        #self.tagged_data = self.text_preprocessing()

        # parameters
        self.epochs= int(epochs)
        self.embedding_size = int(embedding_size)
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = int(min_count)
        self.dm = int(dm)
        self.model_name = model_name

    def text_preprocessing(self) :
        '''
        Do not use this method 
        '''
        return [TaggedDocument(words=nltk.word_tokenize(word.lower()), tags=[str(index)]) for index, word in enumerate(self.document)]

    def train(self) :
        self.doc2vec_model = Doc2Vec(vector_size=self.embedding_size, alpha=self.alpha, min_alpha=self.min_alpha, min_count=self.min_count, dm=self.dm)
        self.doc2vec_model.build_vocab(self.document)

        for epoch in range(self.epochs) :
            if epoch % 10 == 0 :
                print('Epochs : {}'.format(epoch))
            self.doc2vec_model.train(self.document, 
                                total_examples=self.doc2vec_model.corpus_count,
                                epochs=self.doc2vec_model.epochs)
            self.doc2vec_model.alpha -= 0.0002
            
        self.doc2vec_model.save(self.model_name)
        print("Doc2vec model saved")
#parameter
epochs=100
embedding_size=16
alpha=0.001
min_alpha=0.025
min_count=1
dm=1
model_name="test"

model= Doc_to_vec(document=processed_data, epochs=epochs, embedding_size=embedding_size, alpha=alpha, model_name= model_name)
model.train()

tokenized_text = [nltk.regexp_tokenize(doc.lower(),'[A-Za-z]+') for doc in data_concat]
embedding_vectors = [model.doc2vec_model.infer_vector(text) for text in tokenized_text]

#########################################################################################################################################################
#load data
import pickle
import nltk
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/0502_repo_71.pickle"
with open(file_path,"rb") as fr:
    data = pickle.load(fr)
###############################################################################################################################################    
#doc2vec 실행
from nltk.corpus import stopwords

def apply_stop_words(tokenized_text):
    stop_words = stopwords.words('english') 
    stop_words.extend(["is", "a","in","an","of","the","to","for","and","from"]) #필요없는 단어 추가
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  

def text_preprocessing(clean_data):
   # data = apply_stop_words(tokenized_text)
   # clean_data = []
   # for doc in data:
   #     clean_data.append(" ".join(doc))
        
    return [TaggedDocument(words=nltk.regexp_tokenize(word.lower(),'[A-Za-z]+'), tags=[str(index)]) for index, word in enumerate(clean_data)]
"""

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
#load data
import pickle
import nltk
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/0502_repo_69.pickle"
with open(file_path,"rb") as fr:
    data = pickle.load(fr)
data_concat= [str(row['repo']) +" "+ str(row['topics']).replace("#"," ")+" "+ str(row['description']) for idx, row in data.iterrows()]# 토픽 주석표시 제거
processed_data = text_preprocessing(data_concat)

#doc2vec
#########
max_epochs = 100

model = Doc2Vec(
    window=5,
    size=39168,
    alpha=0.05, 
    min_alpha=0.01,
    min_count=2,
    dm =1,
    negative = 5,
    seed = 9999)
  
model.build_vocab(processed_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(processed_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    


similar_doc = model.docvecs.most_similar('1')
print(similar_doc)

tokenized_text = [nltk.regexp_tokenize(doc.lower(),'[A-Za-z]+') for doc in data_concat]
embedding_vectors = [model.infer_vector(text) for text in tokenized_text]

#save data
##################################################################################################################

import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(doc2vec).pickle"
with open(file_path,"wb") as fw:
    pickle.dump(embedding_vectors, fw)   