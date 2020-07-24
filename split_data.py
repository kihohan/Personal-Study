import sys
import random
import ir
import time

start = time.time()

data_dir = 'data/'
mall_file_base =sys.argv[1] if len(sys.argv) >= 2 else 'danawaMall_'
master_file = sys.argv[2] if len(sys.argv) >= 3 else data_dir+'danawa_mws_master_item'
test_ratio = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.2

train_file = data_dir+'train_danawa_all'
test_file = data_dir+'test_danawa_all'

train_fh = open(train_file, 'w')
test_fh = open(test_file, 'w')

count = 0
fh = open(master_file, 'r')
for line in fh:
    count += 1
    if(count != 1):
        dataList =line.strip().split('\t')
        item_id, goods_name, option_name, label1, label2 = ir.extractData2(dataList)
        if((label1 != '기타' and label1 != '') or (label2 != '기타' and label2 != '')):
            rand_num = random.random()
            if(rand_num <= test_ratio):
                test_fh.write(item_id+'\n')
            else:
                train_fh.write(item_id+'\n')
fh.close()
train_fh.close()
test_fh.close()

run_time =time.time()-start
print("running time :",run_time, "seconds, or",run_time/60, "minutes, or", run_time/3600, 'hours' )