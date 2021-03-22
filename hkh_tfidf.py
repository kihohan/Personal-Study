import re
import time
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_pred(cate, test):
    def pre_processing(text):
        text = text.upper()
        # remove all words inside of (). eg, (해외구매)
        text2 = re.sub(r'\([^)]*\)', '', text)
        # remove all words inside of [], eg, [KB국민카드 1% 청구할인]
        text3 = re.sub(r'\[[^\]]*\]', '', text2)
        cleaned_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text3)
        result = re.sub('  ', ' ', cleaned_text)
        return result

    dt = pd.read_pickle('df_danawa.pk')

    dt['pre_goods_name'] = dt['mall_goods_name'].apply(pre_processing)

    df = dt[dt['master_tag'] == cate].reset_index(drop = True)


    my_stopwords = ['골라담기','브랜드','상세','설명','참조','인기''증정','세트','특가','초특가','본사','세트상품','포장','제조',
                '상세설명','증정','기획''본사직영','없음','행사상품','발송','선택','이상','본사공식','공식점','박스']

    tfidf = TfidfVectorizer(
        max_features = 5000,
        stop_words = my_stopwords,
        tokenizer = lambda x:x.split(' '),
        min_df = 10
        )

    meta_file = df['pre_goods_name'].to_list()
    meta_brand = df['brand'].to_list()
    # meta_brand = [v + '_' + str(random.random()) for v in df['brand'].to_list()]
    dct = dict (zip(meta_file, meta_brand))

    test = test
    pre = pre_processing(test)
    # 검색 데이터와 후보군 데이터를 합침
    meta_file.insert(0, pre)
    dct[pre] = ''
    # tfidf matrix를 만듬
    data_tfidf = tfidf.fit_transform(meta_file)
    similarities =  data_tfidf.toarray()[0] * data_tfidf[1:,].T

    y = dict (zip(meta_file[1:], similarities))
    x = sorted(y.items(), key = lambda x: x[1], reverse = True)
    x_df = pd.DataFrame(x, columns=['goods_name','score'])
    x_df['brand'] = x_df['goods_name'].map(dct)
    z = dict(x_df.groupby('brand')['score'].max())
    print (sorted(z.items(), key = lambda x: x[1], reverse = True)[:3])
#    return sorted(z.items(), key = lambda x: x[1], reverse = True)[:3]

if __name__ == "__main__":
    start = time.time()
    cate = input('cateogry:')
    test = input('goods_name:')
    print ('=' * 100)
    print ('caculate tfidf & cosine similarities..........')
    tfidf_pred(cate, test)
    run_time = float('{:0.2f}'.format(time.time() - start))    
    print("running time :",run_time, "seconds, or",run_time/60, "minutes, or", run_time/3600, 'hours' )
