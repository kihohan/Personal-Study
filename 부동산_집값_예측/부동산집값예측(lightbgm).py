#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


os.chdir('/home/kiho/다운로드/data')


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('submission.csv')


# In[4]:


train.head()


# In[5]:


from matplotlib import rc, font_manager
rc('font', family = 'NanumGothic')
sns.countplot(train['city'])


# In[6]:


train.groupby(['city'])['transaction_real_price'].mean()


# In[7]:


train['dong'].nunique()


# In[8]:


train.groupby(['dong'])['transaction_real_price'].mean().sort_values(ascending = False)[:10].plot.bar()


# In[9]:


# mean-encoding: 연속형 범주를 평균값으로 대체하는 인코딩방식
train = pd.merge(train,train.groupby(['dong'])['transaction_real_price'].mean().reset_index(name='dong_price_mean'),on=['dong'],how='left')
test = pd.merge(test,train.groupby(['dong'])['transaction_real_price'].mean().reset_index(name='dong_price_mean'),on=['dong'],how='left')

train = pd.merge(train,train.groupby(['dong']).size().reset_index(name='dong_count'),on=['dong'],how='left')
test = pd.merge(test,train.groupby(['dong']).size().reset_index(name='dong_count'),on=['dong'],how='left')

train['dong_price_mean'] = (train['dong_count']*train['dong_price_mean']-train['transaction_real_price'])/(train['dong_count']-1)
train['dong_count'] = train['dong_count']-1


# In[10]:


plt.scatter(train['exclusive_use_area'],train['transaction_real_price'])
print('corr: ', train['exclusive_use_area'].corr(train['transaction_real_price']))


# In[11]:


# transaction_year_month(실거래 발생 년월)에서 transaction_year(실거래 발생 년)만 추출
train['transaction_year'] = train['transaction_year_month'].astype(str).str.slice(0,4).astype(int)
test['transaction_year'] = test['transaction_year_month'].astype(str).str.slice(0,4).astype(int)


# In[12]:


# transaction_year(실거래 발생 년)에서 year_of_completion(준공년도)를 빼, 실거래 발생 시 아파트의 '나이'라는 변수를 추출한다.
train['apartment_age'] = train['transaction_year']-train['year_of_completion']
test['apartment_age'] = test['transaction_year']-test['year_of_completion']


# In[13]:


sns.lmplot('apartment_age','transaction_real_price',
           data=train.sample(10000),hue='city',order=2,scatter_kws={'alpha':0.1})


# In[14]:


'''sns.lmplot에서 hue를 city로 설정해서 도시별로 age와 실거래가의 관계를 살펴본다. order를 2로 지정해 회귀식의 차원을 2차원으로 늘림으로써 age와 실거래가의 비선형적인 관계를 살펴본다.

회귀식을 보면, age가 0부터 30까지는 실거래가가 감소하는 경향을 보이는 반면 30년이 넘어가면서 실거래가가 급상승한다.

아파트의 재건축 연한이 현재 대한민국에서 30~40년으로 설정되어 있기 때문인 것으로 보인다.

해당 인사이트를 머신러닝에 반영하기 위해 is_rebuild라는 변수를 생성한다(재건축 임박인지 아닌지 여부를 판가름한다)'''

train['is_rebuild']=(train['apartment_age']>=30).astype(int)
test['is_rebuild']=(test['apartment_age']>=30).astype(int)


# In[15]:


# transaction_year_month(실거래 발생 년월)에서 transaction_month(실거래 발생 월)만 추출
train['transaction_month']=train['transaction_year_month'].astype(str).str.slice(4,).astype(int)
test['transaction_month']=test['transaction_year_month'].astype(str).str.slice(4,).astype(int)


# ### 실거래 발생 월별로 실거래가에 유의미한 차이가 있는지 살펴본다.

# In[16]:


train.groupby(['transaction_month'])['transaction_real_price'].mean().plot()


# In[17]:


sns.boxplot(train['transaction_month'],train['transaction_real_price'],showfliers=False)


# In[18]:


import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df = pd.DataFrame(train, columns=['transaction_month', 'transaction_real_price'])    
model = ols('transaction_real_price ~ C(transaction_month)', df).fit()

print(anova_lm(model))
'''일원분산분석 결과 : F=227.600948, p=0.0

P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미하게 차이난다.

month에 따른 실거래가의 차이가 유의미하다!'''


# In[19]:


sns.lmplot('floor','transaction_real_price',data=train.sample(1000),order=2,scatter_kws={'alpha':0.1})
'''order를 2로 지정해 회귀식의 차원을 2차원으로 늘림으로써 층과 실거래가의 비선형적인 관계를 살펴본다. 1층부터 20층까지는 실거래가가 대체적으로 비슷하지만, 그 이후부터는 실거래가의 크기가 급격하게 커지는 경향성을 발견할 수 있다. 아파트에서의 '조망권' 때문인것으로 보인다.
'''


# In[20]:


train.groupby(['dong']).apply(lambda x: x.loc[x['floor']>=30]['transaction_real_price'].mean()-x.loc[x['floor']<=10]['transaction_real_price'].mean()).sort_values(ascending=False)[:10].plot.bar()


# In[21]:


'''해석: 한강 근처는 아무래도 강에 대한 조망권 때문에, 층이 높을수록 가격이 급격하게 비싸지는 경향이 있을 것이다.'''
train.groupby(['dong']).apply(lambda x: x.loc[x['floor']>=30]['transaction_real_price'].mean()-x.loc[x['floor']<=10]['transaction_real_price'].mean()).sort_values(ascending=False)[:10]


# In[22]:


train['hangang'] = train['dong'].isin(['성수동1가','삼성동','이촌동','공덕동','서교동','한강로3가','목동']).astype(int)
test['hangang'] = test['dong'].isin(['성수동1가','삼성동','이촌동','공덕동','서교동','한강로3가','목동']).astype(int)


# ### feature engineering 결과물을 이용한 머신러닝 모델 만들기

# In[23]:


input_var=['exclusive_use_area', 'year_of_completion','floor','dong_price_mean', 'dong_count',
       'transaction_year', 'apartment_age', 'is_rebuild', 'transaction_month','hangang','city']


# In[24]:


train['city'] = train['city'].map({'서울특별시':1,'부산광역시':0})
test['city'] = test['city'].map({'서울특별시':1,'부산광역시':0})


# In[25]:


import lightgbm as lgb
import time


# In[26]:


from sklearn.model_selection import KFold


# In[30]:


#https://gorakgarak.tistory.com/1285
folds = KFold(n_splits = 5,shuffle = True,random_state = 15)

param = {'num_leaves': 100,
         'min_data_in_leaf': 15, 
         'objective':'regression',
         'max_depth': 6,
         'learning_rate': 0.1,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "device": "gpu"}


# In[32]:


predictions_ops = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train['transaction_real_price'].values)):
    trn_data = lgb.Dataset(train[input_var].iloc[trn_idx],label=train['transaction_real_price'].iloc[trn_idx])#,weight=game_by_game.loc[(game_by_game['year']<year)&(game_by_game['AB']>0)]['AB'].iloc[trn_idx])
    val_data = lgb.Dataset(train[input_var].iloc[val_idx],label=train['transaction_real_price'].iloc[val_idx])#,weight=game_by_game.loc[(game_by_game['year']<year)&(game_by_game['AB']>0)]['AB'].iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds = 200)


    predictions_ops += clf.predict(test[input_var], num_iteration=clf.best_iteration)

predictions_ops/=5

s = time.time()
print("Spent {:1f} Seconds".format(time.time() - s))


# In[ ]:


# http://blog.manugarri.com/note-to-self-installing-lightgbm-in-ubuntu-18-04/
#pd.DataFrame({'transaction_id':test['transaction_id'],'transaction_real_price':predictions_ops}).to_csv("my_first_submission.csv",index=False)

