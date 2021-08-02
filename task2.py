import pandas as pd
import tensorflow.contrib.keras as kr
from sklearn.metrics import  accuracy_score
import sklearn
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from cnn import TextCNN
pd.set_option('max_columns',None)
path_train=r'.\train.tsv'
path_test=r'.\test.tsv'
data_train=pd.read_csv(path_train,sep='\t').head(60000)
data_test=pd.read_csv(path_test,sep='\t').head(10000)
index_seq=list(range(1,len(data_train)))
random.seed(6)
random.shuffle(index_seq)
index_train=index_seq[:int(len(index_seq)*0.75)]
index_val=index_seq[int(len(index_seq)*0.75):]
#获取句子

cv=CountVectorizer()
x_train=data_train['Phrase'].values.tolist()
x_train=cv.fit_transform(x_train).toarray()
x_val=[x_train[index] for index in index_val]
x_train=[x_train[index] for index in index_train]
y_train=data_train['Sentiment'].values.tolist()
y_val=[y_train[index] for index in index_val]
y_train=[y_train[index] for index in index_train]

#textcnn
# 将单词tokenizer化
tokenizer = kr.preprocessing.text.Tokenizer(num_words=2000, oov_token='OOV')
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
# 将句子长度转为一致
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_val)
padded_val = kr.preprocessing.sequence.pad_sequences(sequences_train, maxlen=50)
padded_test = kr.preprocessing.sequence.pad_sequences(sequences_val, maxlen=50)

tf.reset_default_graph()
m=[]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = TCNNConfig()
config.vocab_size=2000
keep_prob=config.keep_prob
model=TextCNN(config)
#将paded分割为batches，batch_size=64
def split_into_batch(data):
    num_total=len(data)
    batches=[]
    if num_total%64==0:
        batch_num=num_total/64
    else:
        batch_num=int(num_total/64)+1
    for i in range(batch_num):
        if 64*(i+1)<num_total:
            batch=data[(64*i):64*(i+1)]
            batches.append(batch)
        else:
            batch=data[(64*i):]
            batches.append(batch)
    return batches
batches_x=split_into_batch(data=padded_train)
batches_y=split_into_batch(data=y_train)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        for k in range(len(batches_x)):
            input_x=np.array(batches_x[k])
            input_y=batches_y[k]
            sess.run(model.optim,
                     feed_dict={"input_x:0": input_x, "input_y:0": input_y,
                                "keep_prob:0": config.keep_prob})
            acc = sess.run(model.acc,
                           feed_dict={"input_x:0": input_x, "input_y:0": input_y,
                                      "keep_prob:0": config.keep_prob})
        m.append(acc)
    acc = sess.run(model.acc,
                   feed_dict={"input_x:0": padded_val, "input_y:0": y_val,
                              "keep_prob:0": config.keep_prob})
    all_acc.append(acc)
print('测试集准确率，',all_acc)