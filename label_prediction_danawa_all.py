import sys
import math
import ir
import time

start = time.time()

target = sys.argv[1] if len(sys.argv)>= 2 else 'brand'
data_postfix = sys.argv[2] if len(sys.argv)>= 3 else 'danawa_all'
good_name_weight = float(sys.argv[3]) if len(sys.argv)>=4 else 1.0
option_weight = float(sys.argv[4]) if len(sys.argv)>=5 else 1.0

data_dir = 'data/'
out_dir = 'out/'
result_dir = 'result/'
# output files
goods_profile_file = sys.argv[5] if len(sys.argv)>=6 else out_dir+"test_goods_profile_"+target+"_"+data_postfix;
result_file = sys.argv[6] if len(sys.argv)>=7 else result_dir+"result_recom_"+target+"_"+data_postfix;
# input files
master_file = data_dir+"danawa_mws_master_item"
mall_file_base = "danawaMall_"
stopwords_file = data_dir+"stopwords"
test_file = data_dir+"test_"+data_postfix
df_file = out_dir+"df_"+target+"_train_"+data_postfix
tfidf_file = out_dir+"tfidf_"+target+"_train_"+data_postfix


# get the training item set
# test_set[item_id] = 1
test_set = {}
fh = open(test_file, 'r')
for line in fh:
    test_set[line.strip()] = 1
fh.close()


# storing stopwords
# stopwords[term] = 1
stopwords = {}
fh = open(stopwords_file, 'r')
for line in fh:
    stopwords[line.strip()] = 1
fh.close()


# storing model
# model[label][term] = normalized_tfidf_value
# index[term][label] = 1
# available[term] = 1
model, index, availables = {}, {}, {}
fh = open(tfidf_file, 'r')
for line in fh:
    label, term, tfidf, ntfidf = line.strip().split('\t')
    ir.add_item_into_nested_dicionary(model,label,term,float(ntfidf))
    ir.add_item_into_nested_dicionary(index,term,label,1)
    availables[term] = 1
fh.close()
print('The model is loaded!')


# storing df list. too frequent terms are filtered out here
# idf[term] = idf_value
idfs = {}
fh = open(df_file, 'r')
for line in fh:
    term, df, idf = line.strip().split('\t')
    if term in availables:
        idfs[term] = float(idf)
fh.close()
print('df list is loaded!')


# compute tf value of each word in the specific brand/category
# tf[label][term] = tf_value
# labels[item_id][label] = 1 (lable is either brand or maker)
tfs, labels, titles,  count = {}, {}, {}, 0
fh = open(master_file, 'r')
# read only one line. use small memory comparing with readlines()
for line in fh:
    count += 1
    # ignore the comment line(first line)
    if(count != 1):
        datalist = line.strip().split('\t')
        item_id, goods_name, option, brand, maker = ir.extractData2(datalist)
        # if the item is in the train_set, proceed it
        if(item_id in test_set):
            # count term frequency
            ir.getTF(goods_name, good_name_weight, item_id, stopwords, tfs)
            if (option != ''):
                ir.getTF(option, option_weight, item_id, stopwords, tfs)
            # store item info (labels and titles) for the result files
            titles[item_id] = goods_name
            dic = {}
            if(brand != "기타" and brand != ''):
                dic[brand] = 1
            if (maker != "기타" and maker != ''):
                dic[maker] = 1
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
            if(item_id in test_set):
                for label, temp in labels[item_id].items():
                    ir.getTF(goods_name, good_name_weight, item_id, stopwords, tfs)
    fh.close()
    print(mall_file+" is done!")


# generate test item profile
# profiles[item_id][term] = normalized_tfidf_value
fh = open(goods_profile_file, 'w')
profiles = {}
for item_id, dic in tfs.items():
    tfidfs, norm = {}, 0
    output_head = item_id + '\t' + titles[item_id] + '\t'
    fh.write(output_head)
    for term, tf in dic.items():
        if term in idfs:
            tfidfs[term] = tf * idfs[term]
            norm += pow(tfidfs[term], 2)
    # normalize tfidf value
    sqrt_norm = math.sqrt(norm)
    for term, tfidf  in tfidfs.items():
        tfidfs[term] /= sqrt_norm
        fh.write(term + ":%f, " % tfidfs[term])
    profiles[item_id] = tfidfs
    fh.write("\n")
fh.close()
print('test item profiles are generated!')

# start prediction
total, equal, unequal = 0,0,0
# test result file
fh = open(result_file, 'w')
for item_id, profile in profiles.items():
    # prepare for printing output.
    output_head = item_id + '\t' + titles[item_id] + '\t'
    # find the best candidate among available labels
    cands = {}
    for term, ntfidf in profile.items():
        # due to index, we don't have to compute similarity between the product and all available labels
        for label, value in index[term].items():
            try:
                cands[label] += ntfidf * model[label][term]
            except KeyError:
                cands[label] = ntfidf * model[label][term]
    total += 1
    # write the result only if there exists at least one candidate
    if(len(cands) > 0):
        fh.write(output_head)
        for label in labels[item_id]:
            fh.write(label+', ')
        count = 0
        for cand, score in sorted(cands.items(), key=lambda x: x[1], reverse=True):
            count += 1
            # the best candidate is ..
            if(count == 1):
                if(cand in labels[item_id]):
                    equal += 1
                    output = "\t"+cand+"\t%f\tcorrect\t" % score
                else:
                    unequal += 1
                    output = "\t"+cand+"\t%f\twrong\t" % score
                fh.write(output)
            elif(count <= 3):
                output = cand+":%f, " % score
                fh.write(output)
            if(count == 3):
                break
        fh.write('\n')
# print the test statistics
output = "total:%d, equal:%d, unequal: %d, no candidate: %d, hit ratio: %f\n" \
         %(total,equal,unequal,total-equal-unequal,equal/total)
fh.write(output)
print(output)
fh.close()

run_time =time.time()-start
print("running time :",run_time, "seconds, or",run_time/60, "minutes, or", run_time/3600, 'hours' )