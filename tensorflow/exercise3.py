import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

#rolls = tf.random_uniform([10, 2], minval=1, maxval=7, dtype=tf.int32)
#roll_sums = tf.add(tf.slice(rolls, [0, 0], [10, 1]), tf.slice(rolls, [0, 1], [10, 1]))

first_rolls = tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)
second_rolls = tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32)

roll_sums = tf.add(first_rolls, second_rolls)
matrix_result = tf.concat([first_rolls, second_rolls, roll_sums], axis=1)
print(matrix_result.numpy())

