#!/usr/bin/env python
# coding: utf-8

# In[1]:


# How to check if the code is running on GPU or CPU?
# 아래와 같이 tensorflow의 device_lib이라는 함수를 통해 현재 사용가능한 하드웨어 디바이스들의 리스트를 볼 수 있습니다. 
# 여기서 name을 통해 해당 하드웨어에 접근할 수 있습니다
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[2]:


# How to check if Keras is using GPU?
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[4]:


#계산량이 적은 것들은 첫번쨰 cpu에 할당
with tf.device('/cpu:0'):
    X = tf.placeholder(tf.float32, [None, SIZE_INPUT])
    Y = tf.placeholder(tf.floats32, [None, SIZE_OUTPUT])
#계산량이 많은 건 첫번째 gpu에 할당
with tf.device('/gpu:0'):
    COST_VECTOR = tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y)
    COST = tf.reduce_mean(COST_VECTOR)
    OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE).minimize(COST)
#케라스의 경우 GPU
import keras.backend.tensorflow_backend as K
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(Dense(512, input_dim=28*28, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train_cat, batch_size=128*4, epochs=10, verbose=1, validation_split=0.3)
#케라스의 경우 cpu
    with K.tf.device('/cpu:0'):
    model = Sequential()
    model.add(Dense(512, input_dim=28*28, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train_cat, batch_size=128*4, epochs=10, verbose=1, validation_split=0.3)


# ## 단어 수준의 원-핫 인코딩하기

# In[6]:


import numpy as np


# In[2]:


samples = ['the mountain is here.','the beach is over there']


# In[3]:


token_index = {}


# In[4]:


for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1


# In[5]:


print (token_index)


# In[6]:


max_length = 7


# In[7]:


results = np.zeros(shape = (len(samples),
                  max_length,
                  max(token_index.values()) + 1))


# In[8]:


print (results)


# In[9]:


for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.


# In[11]:


print (results)


# ## 문자 수준 원-핫 인코딩하기

# In[28]:


import string


# In[29]:


samples2 = ['the mountain is here.','the beach is over there']
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))


# In[31]:


max_length = 50
results = np.zeros((len(samples2), max_length, max(token_index.values()) + 1))


# In[32]:


print (results)


# In[34]:


for i, sample in enumerate(samples2):
    for j, character in enumerate(samples2):
        index = token_index.get(character)
        results[i,j,index] = 1


# In[35]:


print (results)


# ## 케라스를 사용한 단어 수준의 원-핫 인코딩하기(원-핫 해싱)

# In[36]:


from keras.preprocessing.text import Tokenizer


# In[37]:


samples3 = ['the mountain is here.','the beach is over there']


# In[38]:


tokenizer = Tokenizer(num_words = 1000) #가장 빈도가 높은 1,000개의 단어만 선택하도록 Tokenizer객체를 만듭니다
tokenizer.fit_on_texts(samples3)


# In[39]:


sequences = tokenizer.texts_to_sequences(samples)


# In[40]:


one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary')


# In[41]:


word_index = tokenizer.word_index #계산된 단어 인텍스를 구합니다


# In[55]:


print ('%s개의 고유한 토큰을 찾았습니다.' % len(word_index))
print ('{0}개의 고유한 토큰을 찾았습니다.'.format(len(word_index)))


# In[ ]:


# 해싱함수: 1.체인잉: 리스트같은 자료 구조를 이용해서 옆에 붙이는 방식
#          2.linear Probing: 다음 버켓에 그냥 넣어준다.
#          3.리사이징: 테이블의 크기를 늘려서 재정렬을 한다.


# ## 해싱 기법을 사용한 단어 수준의 원-핫  인코딩

# In[20]:


samples4 = ['the mountain is here.','the beach is over there']


# In[21]:


dimensionality = 1000 #단어를 크기가 1,000인 백터로 저장한다.
max_length = 10


# In[23]:


results = np.zeros((len(samples4), max_length, dimensionality))
for i, sample in enumerate(samples4):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality #abs(): 숫자의 절대값을 반환해준다.
        results[i, j, index] = 1.
#해시 함수(hash function)는 임의의 길이의 데이터를 고정된 길이의 데이터로 매핑하는 함수이다. 


# In[24]:


print (results)


# In[6]:


import pandas as pd
dt = pd.read_csv('/home/kiho/다운로드/TaxiMach_Link_Dataset_Full_201709.txt')


# In[7]:


dt.head()


# In[10]:


dt['T_Link_ID'].nunique()


# In[ ]:




