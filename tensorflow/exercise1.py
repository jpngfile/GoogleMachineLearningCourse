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

two = tf.constant(2, dtype=tf.int32)
primes_squared = tf.pow(primes, two)
just_below_primes_squared = tf.subtract(primes_squared, ones)
print()
print("just_below_primes_squared:", just_below_primes_squared)

