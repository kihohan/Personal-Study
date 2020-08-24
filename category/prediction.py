import re
import pandas as pd
import pickle
import numpy as np

import sentencepiece as spm
from tensorflow import keras
from keras.preprocessing import sequence

def clean_spm(lst):
    def clean_text(text):
        return re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
    def clean_num(text):
        return re.sub('\d+', '', text)
    def _del(text):
        return text.replace('▁','')
    a = [clean_text(x) for x in lst] 
    b = [clean_num(x) for x in a] 
    c = [_del(x) for x in b]
    d = [x for x in c if len(x) != 0]
    e = ['즉석죽' if x=='죽' else x for x in d]
    f = ['껌껌' if x=='껌' else x for x in e]
    g = ['Tea' if x=='티' else x for x in f]
    h = [x for x in g if len(x) != 1]
    return h

def prediction(text):
    # label_transform
    _class = ['건강보조식품', '견과', '과일', '과자', '김치', '냉장/냉동/간편식', '당류', '떡', '만두',
            '면류', '빵', '상품밥', '생수', '수산물/건어물', '쌀/잡곡', '아이스크림', '우유', '음료/주류',
            '정육', '조미김', '조미료/장류', '즉석죽', '차', '채소', '치즈', '커피', '탄산음료', '통조림',
            '햄']
    _num = [x for x in range(len(_class))]

    mapping_dct = dict(zip(_num,_class))
    # spm_load
    sp = spm.SentencePieceProcessor()
    sp.Load('cate_food_spm.model')
    # tkn_load
    with open('cate_food_tkn.pickle', 'rb') as handle:
        tkn = pickle.load(handle)
    # model_load
    classification_model = keras.models.load_model('cate_food_model.h5')
    # prediction
    # text = '쉐프드 쉐푸드 명란오일파스타 285g 6종 즉석식품 냉동식품'
    pre = ' '.join(clean_spm(sp.encode_as_pieces(text)))
    print ('pre:',pre)
    t = sequence.pad_sequences(tkn.texts_to_sequences([pre]), maxlen = 50)
    Preds = classification_model.predict(t)

    p = [np.argmax(x) for x in Preds]
    prob = [np.max(x) for x in Preds]

    pred = pd.Series(p).map(mapping_dct)
    print('pred:', pred[0])
    print('prob: {0:.2f}%'.format(prob[0]))