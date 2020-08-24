import re

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