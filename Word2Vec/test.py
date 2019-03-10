## source https://github.com/ujhuyz0110/wrd_emb/blob/master/word2vec_skipgram_medium_v1.ipynb

import numpy as np
from dataprep import tokenize, mapping, generate_training_data
from skipgram import skipgram_model_training
from forwardpropagation import forward_propagation

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

#doc = "After the deduction of the costs of investing, " \
#      "beating the stock market is a loser's game."

doc = open("input_data.txt", 'r').read()

## TODO: CLEAN THIS DATA

tokens = tokenize(doc)
for word in tokens:
    if word in en_stops:
        tokens.remove(word)


word_to_id, id_to_word = mapping(tokens)
X, Y = generate_training_data(tokens, word_to_id, 3)
vocab_size = len(id_to_word)
m = Y.shape[1]
# turn Y into one hot encoding
Y_one_hot = np.zeros((vocab_size, m))
Y_one_hot[Y.flatten(), np.arange(m)] = 1


paras = skipgram_model_training(X, Y_one_hot, vocab_size, 50, 0.05, 5000, batch_size=128, parameters=None, print_cost=True)

X_test = np.arange(vocab_size)
X_test = np.expand_dims(X_test, axis=0)
softmax_test, _ = forward_propagation(X_test, paras)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]

for input_ind in range(vocab_size):
    input_word = id_to_word[input_ind]
    output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
    print("{}'s neighbor words: {}".format(input_word, output_words))


