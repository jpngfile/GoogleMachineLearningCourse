import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
    print("TF imported with eager execution!")
except ValueError:
    print("TF already imported with eager execution!")

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)

ones = tf.ones([6], dtype=tf.int32)
print("ones:", ones)

just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes:", just_beyond_primes)

twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print(some_matrix)
print("\nvalue of some_matrix is:\n", some_matrix.numpy())

# A scalar (0-D tensor)
scalar = tf.zeros([])

# A vector with 3 elements
vector = tf.zeros([3])

# A matrix with 2 rows and 3 columns
matrix = tf.zeros([2, 3])

print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.numpy())
print('vector has shape', vector.get_shape(), 'and value:\n', vector.numpy())
print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.numpy())

# Broadcasting
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes:", primes)

ones = tf.ones([1], dtype=tf.int32)
print("ones:", ones)

just_beyond_primes = tf.add(primes, ones)
print("just_beyond_primes:", just_beyond_primes)

two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)


