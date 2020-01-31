from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(
    num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction,
                            income_prediction, gender_prediction])

model.summary()

model.compile(
    optimizer='rmsprop',
    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.25, 1., 10.]
)

model.compile(
    optimizer='rmsprop',
    loss={
        'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'
    },
    loss_weights={
        'age': 0.25,
        'income': 1.,
        'gender': 10.
    }
)
