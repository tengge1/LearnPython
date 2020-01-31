from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import sys

f = open('nietzsche.txt', 'r')
text = f.read().lower()
f.close()
print('Corpus length:', len(text))

# 将字符序列向量化

maxlen = 60
step = 12  # 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 构建网络

model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


model.fit(x, y, batch_size=128, epochs=1)  # epochs=60，花时间太长

start_index = random.randint(0, len(text) - maxlen - 1)

generated_text = text[start_index: start_index + maxlen]

print('--- Generating with seed: "' + generated_text + '"')

for temperature in [0.2, 0.5, 1.0, 1.2]:
    print('\n\n------ temperature:', temperature)
    sys.stdout.write(generated_text)
    for i in range(400):
        sampled = np.zeros((1, maxlen, len(chars)))

        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]

        generated_text += next_char
        generated_text = generated_text[1:]

        sys.stdout.write(next_char)
