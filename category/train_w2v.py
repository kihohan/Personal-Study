from gensim.models import Word2Vec

def make_w2v_model(data_series, embedding_dim, model_name):
    sentences = data_series.drop_duplicates().apply(lambda x:x.split(' ')).to_list()
    
    model = Word2Vec(sentences, size = embedding_dim, window = 3, min_count = 3, workers = 32)

    word_vectors = model.wv
    vocabs = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]
    print ('Vocab Size:',len(model.wv.vocab))
    # print (word_vectors.similarity(w1 = '즉석밥', w2 = '햇반'))
    # print (model.wv.most_similar('햇반')[:5])
    filename = model_name + '.txt' 
    model.wv.save_word2vec_format(filename, binary = False)