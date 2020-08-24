import sentencepiece as spm

def make_spm_tokenizer(train_text, vocab_size, model_prefix):
    templates = '--input={} --model_prefix={} --vocab_size={}'
    cmd = templates.format(train_text, model_prefix, vocab_size)
    spm.SentencePieceTrainer.train(cmd)