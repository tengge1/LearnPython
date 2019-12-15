import tensorflow as tf


@tf.function
def add(a, b):
    return a + b


result = add(5, 3)

tf.print(result)
