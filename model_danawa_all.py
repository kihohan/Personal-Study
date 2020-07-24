import sys
import ir
import time

start = time.time()

target = sys.argv[1] if len(sys.argv)>= 2 else 'brand'
data_postfix = sys.argv[2] if len(sys.argv)>= 3 else 'danawa_all'
max_df_ratio = float(sys.argv[3]) if len(sys.argv)>=4 else 0.1
min_df = int(sys.argv[4]) if len(sys.argv)>=5 else 1
good_name_weight = float(sys.argv[5]) if len(sys.argv)>=6 else 1.0
option_weight = float(sys.argv[6]) if len(sys.argv)>=7 else 1.0

data_dir = 'data/'
out_dir = 'out/'
stopwords_file = data_dir+"stopwords"
train_file = data_dir+ 'train_danawa_all'
master_file = data_dir+"danawa_mws_master_item"
mall_file_base = "danawaMall_"
tf_file = out_dir+"tf_"+target+"_train_"+data_postfix
df_file = out_dir+"df_"+target+"_train_"+data_postfix
tfidf_file = out_dir+"tfidf_"+target+"_train_"+data_postfix

# get the training item set
# train[item_id] = 1
train_set = {}
fh = open(train_file, 'r')
for line in fh:
    train_set[line.strip()] = 1
fh.close()

# hash storing stopwords
# stopwords[term] = 1
stopwords = {}
fh = open(stopwords_file, 'r')
for line in fh:
    stopwords[line.strip()] = 1
fh.close()

# compute tf value of each word in the specific brand/category
# tf[label][term] = tf_value
# labels[item_id][label] = 1 (lable is either brand or maker)
tfs, labels, count = {}, {}, 0
fh = open(master_file, 'r')
# read only one line. use small memory comparing with readlines()
for line in fh:
    count += 1
    # ignore the comment line(first line)
    if(count != 1):
        datalist = line.strip().split('\t')
        item_id, goods_name, option, brand, maker = ir.extractData2(datalist)
        # if the item is in the train_set, proceed it
        if(item_id in train_set):
            dic = {}
            if(brand != "기타" and brand != ''):
                dic[brand] = 1
                ir.getTF(goods_name, good_name_weight, brand, stopwords, tfs)
                if(option != ''):
                    ir.getTF(option, option_weight, brand, stopwords,tfs)
            if(maker != "기타" and maker != ''):
                dic[maker] = 1
                ir.getTF(goods_name, good_name_weight, maker, stopwords, tfs)
                if(option != ''):
                    ir.getTF(option, option_weight, maker, stopwords,tfs)
            if(len(dic)>0):
                labels[item_id] = dic
fh.close()
print(master_file+" is done!")

# start processing mall file
for i in range(1,6):
    mall_file = data_dir+mall_file_base+str(i)
    fh =open(mall_file, 'r')
    count = 0
    # read only one line. use small memory comparing with readlines()
    for line in fh:
        count += 1
        # ignore the comment line(first line)
        if(count != 1):
            datalist = line.strip().split('\t')
            # option, label1 and label2 should be '' in the mall_data_file
            item_id, goods_name, option, label1, label2 = ir.extractData2(datalist)
            # if the item is in the train set, proceed it
            if(item_id in train_set):
                for label, temp in labels[item_id].items():
                    ir.getTF(goods_name, good_name_weight, label, stopwords, tfs)
    fh.close()
    print(mall_file+" is done!")

# compute df value of each word
# dfs[term] = df_values
idf, dfs, max_df = ir.getDF(tfs, max_df_ratio)
# compute tfidf value of each word in the specific brand/category
# tfidfs[label][term] = tfidf_value
# ntfidfs[label][term] = normalized_tfdif_value
tfidfs, ntfidfs = ir.getTFIDF(tfs,dfs,idf,max_df,min_df)

#generate df_file
fh = open(df_file, 'w')
for word,df_value in sorted(dfs.items(), key=lambda x:x[1], reverse=True):
    line = word+"\t%d\t%f\n" %(df_value,idf[word])
    fh.write(line)
fh.close()

#generate tf_file
fh = open(tf_file, 'w')
for label, item in sorted(tfs.items()):
    for word, tf_value in sorted(item.items(), key=lambda x:x[1], reverse=True):
        if(dfs[word] <= max_df and dfs[word] >= min_df):
            line =label+"\t"+word+"\t%d\n" % tf_value
            fh.write(line)
fh.close()

#generate tfidf_file
fh = open(tfidf_file, 'w')
for label, item in sorted(tfidfs.items()):
    for word, tfidf_value in sorted(item.items(), key=lambda x:x[1], reverse=True):
        line =label+"\t"+word+"\t%f\t%f\n" %(tfidf_value, ntfidfs[label][word])
        fh.write(line)
fh.close()
run_time =time.time()-start
print("running time :",run_time, "seconds, or",run_time/60, "minutes, or", run_time/3600, 'hours' )