import pickle

word_dict = pickle.load(open('./train_model/code_word.pkl', 'rb'))
chars     = word_dict.values()
chars_list= list(chars)
test_cnn  = '钟国辉老师'

for c in test_cnn:
    print(c, (c in chars_list))