import re
import math

def extractData(datalist, target, data_postfix):
    if re.search('coupang', data_postfix):
        item_id, category, good_name, option_name, label = datalist
        return (item_id.strip(), good_name, option_name, label.strip())
    elif re.search('dongwon', data_postfix):
        item_id, option_id, site_name, collect_site, date1, date2, day_of_week,\
        thumbnail, good_name, option_name, quantity, url, brand, sub_brand, brand_type, \
        traverse_category, category_id, cat1, cat2, cat3, cat4, cat5, \
        seller, biz_num, busi_owner, addr, contact, hp_contact,  \
        original_price, discounted_price1, discounted_price2, mileage, manufacturer, \
        delivery_info, delivery_cost, is_rocket_delivery,\
        sell_amount, total_revenue, num_review, avg_score, num_pick, \
        is_promo, is_deal, key, reg_date, master_brand_name = datalist
        new_item_id = item_id+"_"+option_id
        label = cat4 if target == 'category' else master_brand_name
        return (new_item_id, good_name, option_name, label.strip())
    elif re.match('danawa_food', data_postfix):
        id1, id2, category, maker, brand, master_name, good_name, url, master_brand = datalist
        item_id = id1+"_"+id2
        new_good_name, option_name = master_name, good_name
        label = category if target == 'category' else master_brand
        return (item_id, new_good_name, option_name, label.strip())
    else:
        return ('','','','')

def extractData2(datalist):
    # if the data come from the master file, ..
    if (len(datalist) == 21):
        item_id, collect_site, item_num, goods_num, goods_cate, main_goods_cate, \
        prd_name, option_name, maker_name, brand_name, url, lowest_price, tag, add_info, \
        avg_point, total_eval, total_qna, total_news, status, reg_dt, upt_dt = datalist
#        if(brand_name != ''):
        return (item_id, prd_name, option_name, brand_name.strip(), maker_name.strip())
#        elif(maker_name != ''):
#            return (item_id, prd_name, option_name, maker_name.strip())
#        else:
#            return ('','','','')
    # if the data come from the mall file
    elif (len(datalist) == 5):
        mall_id, item_id, mall_name, mall_item_num, prd_name = datalist
        return (item_id, prd_name, '', '', '')
    else:
        return ('','','','','')

def cleanText(text):
    # remove all words inside of (). eg, (해외구매)
    text2 = re.sub(r'\([^)]*\)', '', text)
    # remove all words inside of [], eg, [KB국민카드 1% 청구할인]
    text3 = re.sub(r'\[[^\]]*\]', ' ', text2)
    cleaned_text = re.sub('[^A-Za-z가-힣0-9]', ' ', text3)
    return cleaned_text.strip()


def shouldRemove(word, stopwords):
    # if the word consists of only numbers, ignore it
    if re.match('^[0-9]+$', word):
        return 1
    if word in stopwords:
        return 1
    elif len(word) <= 1:
        return 1
    else:
        return 0


def getTF(text, weight, label, stopwords, tf):
    if not re.match('^\s+$', text):
        cleaned_text = cleanText(text)
        local_tf = {}
        if label in tf:
            local_tf = tf[label]
        for word in re.split('\s+', cleaned_text):
            if not shouldRemove(word, stopwords):
                lowercase = word.lower()
                try:
                    local_tf[lowercase] += weight
                except KeyError:
                    local_tf[lowercase] = weight
        tf[label] = local_tf


def getItemTF(text, weight, stopwords, tf):
    if not re.match('^\s+$', text):
        cleaned_text = cleanText(text)
        for word in re.split('\s+', cleaned_text):
            if not shouldRemove(word, stopwords):
                lowercase = word.lower()
                try:
                    tf[lowercase] += weight
                except KeyError:
                    tf[lowercase] = weight


def getDF(tf, max_df_ratio):
    df = {}
    num_label = 0
    for label, dict in tf.items():
        num_label += 1
        for word, value in dict.items():
            try:
                df[word] += 1
            except KeyError:
                df[word] = 1
    idf = {}
    for word, df_value in df.items():
        idf[word] = math.log(num_label/df_value)
    max_df = num_label * max_df_ratio
    return (idf, df, max_df)


def getTFIDF(tf, df, idf, max_df, min_df):
    tfidf = {}
    norm = {}
    ntfidf = {}
    for label, item in tf.items():
        local_tfidf = {};
        for word, tf_value in item.items():
            if(df[word] >= min_df and df[word] <= max_df):
                local_tfidf[word] = tf_value * idf[word]
                try:
                    norm[label] += pow(local_tfidf[word],2)
                except KeyError:
                    norm[label] = pow(local_tfidf[word], 2)
        tfidf[label] = local_tfidf
    for label, item in tfidf.items():
        local_ntfidf = {};
        for word, tfidf_value in item.items():
            local_ntfidf[word] = tfidf_value / math.sqrt(norm[label])
        ntfidf[label] = local_ntfidf
    return (tfidf, ntfidf)

def add_item_into_nested_dicionary(dict, key1, key2, value):
    local_dict = {}
    if key1 in dict:
        local_dict = dict[key1]
    local_dict[key2] = value
    dict[key1] = local_dict

