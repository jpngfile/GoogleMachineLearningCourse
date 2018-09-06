import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 2])

a_reshaped = tf.reshape(a, [2, 3])
b_reshaped = tf.reshape(b, [3, 1])
matrix_product = tf.matmul(a_reshaped, b_reshaped)
print("Reshaped a matrix:")
print(a_reshaped.numpy())
print("Reshaped b matrix:")
print(b_reshaped.numpy())
print("Matrix product:")
print(matrix_product.numpy())
