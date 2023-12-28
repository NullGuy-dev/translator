# !!! WARNING !!! WARNING !!! WARNING !!!
# open LINKS TO THE DATASET AND TRAINED MODEL file, and download dataset and model there

# importing our lib's and frameworks
import string
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop

# getting our data and them splitting for working
with open("eng_rus.txt", 'r', encoding='utf8') as file:
    data = file.read()
e_r = data.strip().split('\n')
e_r = np.array([i.split('\t') for i in e_r]) # if you would just to try that, you can use this - [:50000, :]

# making lowercase and punctuation deleting for data
e_r[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in e_r[:, 0]]
e_r[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in e_r[:, 1]]

# adding english and russian data lengths
eng_len = []
rus_len = []

for s in e_r[:, 0]:
    eng_len.append(len(s.split()))

for s in e_r[:, 1]:
    rus_len.append(len(s.split()))

def create_tok(data): # function for creating tokenizer
    tok = Tokenizer()
    tok.fit_on_texts(data)
    return tok

def get_out_of_tok(tok, tok_len, texts): # function for getting the sequences from tokenizer
    seq = tok.texts_to_sequences(texts)
    seq = pad_sequences(seq, maxlen=tok_len, padding='post')
    return seq

# creating tokenizer for english part of data and getting its vok size
eng_tok = create_tok(e_r[:, 0])
eng_vok_s = len(eng_tok.word_index) + 1
eng_len = 8

# creating tokenizer for russian part of data and getting its vok size
rus_tok = create_tok(e_r[:, 1])
rus_vok_s = len(rus_tok.word_index) + 1
rus_len = 8

# cutting data for russian part and english parts of this for training NN
# for this we must get sequences from tokenizers
trainX, trainY = get_out_of_tok(rus_tok, rus_len, e_r[:, 1]), get_out_of_tok(eng_tok, eng_len, e_r[:, 0])

# defining model [THIS PARAMS ARE THE BEST, BUT IF YOU CAN CORRECT THIS - WRITE ME]
model = Sequential([
    Embedding(rus_vok_s, 512, input_length=rus_len, mask_zero=True),
    LSTM(512),
    RepeatVector(eng_len),
    LSTM(512, return_sequences=True),
    Dense(eng_vok_s, activation='softmax')
])

model_file_name = "model_eng_rus.h1.fv_28_23" # name for trained model's file

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compiling model by good options
model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=150, batch_size=512, validation_split=0.15, verbose=1) # training model by 150 epochs. And I putted 15% of training data for validation
model.save(model_file_name) # saving model with .h1 type

# model = load_model(model_file_name) # if you need this

def translate_text(model, text, in_tok, in_len, out_tok): # function for using model
    seq = get_out_of_tok([text], in_tok, in_len) # getting seq of input text
    pred_seq = model.predict(seq) # model exploitation with seq
    res = ''
    for ti in np.argmax(pred_seq, axis=-1)[0]: # checking input text
        print(ti)
        if ti == 0: # if it's not right word - stop algorithm
            break
        res += out_tok.index_word[ti] + ' ' # adding word to the result
    return res.strip() # returning result, and space's deleting of the begining and ending of sentence

while True: # just example for model using
    val = input("? ")
    if not val:
        break
    print(translate_text(model, val, rus_tok, rus_len, eng_tok))
