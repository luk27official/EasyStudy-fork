import tensorflow as tf

if __name__ == "__main__":
    print("Hello, TensorFlow!")

    print(tf.__version__)
    print(tf.config.list_physical_devices("GPU"))

    with tf.device("/gpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
        c = tf.matmul(a, b)

        print(c)
