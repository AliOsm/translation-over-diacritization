import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

# Constants

TRAIN = True
USE_DIACS = True
BATCH_SIZE = 256
EPOCHS = 50
EMBEDDINGS_DIM = 64
UNITS = 256

# Helpers

def remove_diacritics(text):
    diacritics_list = ''.join(['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ'])
    return text.translate(str.maketrans('', '', ''.join(diacritics_list)))

def extract_diacritics(text):
    diacritics_list = ''.join(['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ'])
    diacritics = ''
    for char in text:
        if char in diacritics_list:
            diacritics += char
    if diacritics == '':
        diacritics = '<none>'
    return diacritics

# Data Preparing

def create_dataset():
    ar_lines = open('data_dir/ar.bpe.train').read().strip().split('\n')
    for idx in range(len(ar_lines)):
        ar_lines[idx] = '<start> ' + ar_lines[idx].strip() + ' <end>'
    
    en_lines = open('data_dir/en.bpe.train').read().strip().split('\n')
    for idx in range(len(en_lines)):
        en_lines[idx] = '<start> ' + en_lines[idx].strip() + ' <end>'
    
    if USE_DIACS:
        ar_diac_lines = open('data_dir/ar-diac.bpe.train').read().strip().split('\n')
        for idx in range(len(ar_diac_lines)):
            ar_diac_lines[idx] = ' '.join([extract_diacritics(token) for token in ar_diac_lines[idx].split()])
            ar_diac_lines[idx] = '<start> ' + ar_diac_lines[idx] + ' <end>'

        return ar_lines, ar_diac_lines, en_lines
    
    return ar_lines, en_lines

if USE_DIACS:
    ar, ar_diac, en = create_dataset()
else:
    ar, en = create_dataset()

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def load_dataset():
    if USE_DIACS:
        ar_lang, ar_diac_lang, en_lang = create_dataset()
    else:
        ar_lang, en_lang = create_dataset()

    ar_tensor, ar_lang_tokenizer = tokenize(ar_lang)
    en_tensor, en_lang_tokenizer = tokenize(en_lang)
    
    if USE_DIACS:
        ar_diac_tensor, ar_diac_lang_tokenizer = tokenize(ar_diac_lang)
        return ar_tensor, ar_diac_tensor, en_tensor, ar_lang_tokenizer, ar_diac_lang_tokenizer, en_lang_tokenizer

    return ar_tensor, en_tensor, ar_lang_tokenizer, en_lang_tokenizer

if USE_DIACS:
    ar_tensor, ar_diac_tensor, en_tensor, ar_lang, ar_diac_lang, en_lang = load_dataset()
    max_length_ar, max_length_ar_diac, max_length_en = max_length(ar_tensor), max_length(ar_diac_tensor), max_length(en_tensor)
    print(max_length_ar, max_length_ar_diac, max_length_en)
else:
    ar_tensor, en_tensor, ar_lang, en_lang = load_dataset()
    max_length_ar, max_length_en = max_length(ar_tensor), max_length(en_tensor)
    print(max_length_ar, max_length_en)

def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))

print("AR Language; index to word mapping")
convert(ar_lang, ar_tensor[0])
print()
print("EN Language; index to word mapping")
convert(en_lang, en_tensor[0])
if USE_DIACS:
    print()
    print("AR DIAC Language; index to word mapping")
    convert(ar_diac_lang, ar_diac_tensor[0])

BUFFER_SIZE = len(ar_tensor)
steps_per_epoch = len(ar_tensor) // BATCH_SIZE
vocab_ar_size = len(ar_lang.word_index) + 1
vocab_en_size = len(en_lang.word_index) + 1

if USE_DIACS:
    vocab_ar_diac_size = len(ar_diac_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices((ar_tensor, ar_diac_tensor, en_tensor)).shuffle(BUFFER_SIZE)
else:
    dataset = tf.data.Dataset.from_tensor_slices((ar_tensor, en_tensor)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# The Model

if USE_DIACS:
    class Encoder(tf.keras.Model):
        def __init__(self, ar_vocab_size, ar_diac_vocab_size, embedding_dim, units, batch_size):
            super(Encoder, self).__init__()
            self.batch_size = batch_size
            self.units = units
            self.ar_embedding = tf.keras.layers.Embedding(ar_vocab_size,
                                                          embedding_dim,
                                                          embeddings_initializer='glorot_uniform')
            self.ar_diac_embedding = tf.keras.layers.Embedding(ar_diac_vocab_size,
                                                           		 embedding_dim,
                                                          		 embeddings_initializer='glorot_uniform')
            self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units,
                                                                           return_sequences=True,
                                                                           return_state=True,
                                                                           recurrent_initializer='glorot_uniform'))

        def call(self, ar, ar_diac, hidden):
            ar = self.ar_embedding(ar)
            ar_diac = self.ar_diac_embedding(ar_diac)
            output, state_fh, state_fc, state_bh, state_bc = self.lstm(tf.keras.layers.concatenate([ar, ar_diac]), initial_state=hidden)
            state = tf.keras.layers.concatenate([state_fh, state_fc, state_bh, state_bc])
            return output, state

        def initialize_hidden_state(self):
            return [
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units))
            ]
else:
    class Encoder(tf.keras.Model):
        def __init__(self, ar_vocab_size, embedding_dim, units, batch_size):
            super(Encoder, self).__init__()
            self.batch_size = batch_size
            self.units = units
            self.ar_embedding = tf.keras.layers.Embedding(ar_vocab_size,
                                                          embedding_dim,
                                                          embeddings_initializer='glorot_uniform')
            self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units,
                                                                           return_sequences=True,
                                                                           return_state=True,
                                                                           recurrent_initializer='glorot_uniform'))

        def call(self, ar, hidden):
            ar = self.ar_embedding(ar)
            output, state_fh, state_fc, state_bh, state_bc = self.lstm(ar, initial_state=hidden)
            state = tf.keras.layers.concatenate([state_fh, state_fc, state_bh, state_bc])
            return output, state

        def initialize_hidden_state(self):
            return [
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units)),
                tf.zeros((self.batch_size, self.units))
            ]

if USE_DIACS:
    encoder = Encoder(vocab_ar_size, vocab_ar_diac_size, EMBEDDINGS_DIM, UNITS, BATCH_SIZE)
else:
    encoder = Encoder(vocab_ar_size, EMBEDDINGS_DIM, UNITS, BATCH_SIZE)

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, en_vocab_size, embedding_dim, units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units * 2
        self.embedding = tf.keras.layers.Embedding(en_vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(en_vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x)
        state = tf.keras.layers.concatenate([state_h, state_c])
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(vocab_en_size, EMBEDDINGS_DIM, UNITS, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_dir = './bi_without_diac'
if USE_DIACS:
    checkpoint_dir = './bi_with_diac'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# Training

@tf.function
def train_step(sequences, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        if USE_DIACS:
            enc_output, enc_hidden = encoder(sequences[0], sequences[1], enc_hidden)
            en = sequences[2]
        else:
            enc_output, enc_hidden = encoder(sequences[0], enc_hidden)
            en = sequences[1]

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([en_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, en.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(en[:, t], predictions)

            dec_input = tf.expand_dims(en[:, t], 1)

    batch_loss = (loss / int(en.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

if TRAIN:
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, sequences) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(sequences, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# Evaluation

def evaluate(sequences):
    attention_plot = np.zeros((max_length_en, max_length_ar))
    
    for idx in range(len(sequences)):
        sequences[idx] = '<start> ' + sequences[idx] + ' <end>'

    if USE_DIACS:
        ar = list()
        for i in sequences[0].split():
            if i in ar_lang.word_index:
                ar.append(ar_lang.word_index[i])
            else:
                ar.append(0)
        ar = tf.keras.preprocessing.sequence.pad_sequences([ar],
                                                                                                             maxlen=max_length_ar,
                                                                                                             padding='post')
        ar = tf.convert_to_tensor(ar)
        
        ar_diac = list()
        for i in sequences[1].split():
            if i in ar_diac_lang.word_index:
                ar_diac.append(ar_diac_lang.word_index[i])
            else:
                ar_diac.append(ar_diac_lang.word_index['<none>'])
        ar_diac = tf.keras.preprocessing.sequence.pad_sequences([ar_diac],
                                                                                                                        maxlen=max_length_ar_diac,
                                                                                                                        padding='post')
        ar_diac = tf.convert_to_tensor(ar_diac)
    else:
        ar = list()
        for i in sequences[0].split():
            if i in ar_lang.word_index:
                ar.append(ar_lang.word_index[i])
            else:
                ar.append(0)
        ar = tf.keras.preprocessing.sequence.pad_sequences([ar],
                                                                                                             maxlen=max_length_ar,
                                                                                                             padding='post')
        ar = tf.convert_to_tensor(ar)

    result = ''

    hidden = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS)), tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]
    if USE_DIACS:
        enc_out, enc_hidden = encoder(ar, ar_diac, hidden)
    else:
        enc_out, enc_hidden = encoder(ar, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([en_lang.word_index['<start>']], 0)

    for t in range(max_length_en):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                                                                 dec_hidden,
                                                                                                                 enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += en_lang.index_word[predicted_id] + ' '

        if en_lang.index_word[predicted_id] == '<end>':
            return result, sequences, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sequences, attention_plot

def translate(sequences):
    result, sequences, _ = evaluate(sequences)

    return result

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

test_file = 'data_dir/ar.bpe.test'
if USE_DIACS:
    test_file = 'data_dir/ar-diac.bpe.test'

with open(test_file, 'r') as file:
    test_lines = file.readlines()

result = list()
for line in tqdm(test_lines):
    line = line.strip()
    
    if USE_DIACS:
        sequences = list()
        sequences.append(remove_diacritics(line))
        diacss = list()
        for token in line.split():
            diacss.append(extract_diacritics(token))
        sequences.append(' '.join(diacss))
    else:
        sequences = [line]
    
    result.append(translate(sequences))

with open(test_file + '.predictions', 'w') as file:
    file.write('\n'.join(result))
