import tensorflow as tf

x = tf.constant(5.2)
y = tf.Variable([5.0])
y = y.assign([6.0])

z = tf.add(x, y)
with tf.Session() as sess:
    # The initializer is required to use any tf Variables
    initialization = tf.global_variables_initializer()
    print(y.eval())
    print(z.eval())

# Create a graph
g = tf.Graph()

# Establish the graph as the "default" graph
with g.as_default():
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    z = tf.constant(4, name="z_const")
    my_sum = tf.add(x, y, name="x_y_sum")
    more_sum = tf.add(my_sum, z, name="x_y_z_sum")

    with tf.Session() as sess:
        print(more_sum.eval())
