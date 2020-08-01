import tensorflow as tf

def ecanet_layer(self, x): 
        k_size = 3 
        squeeze = tf.reduce_mean(x,[2,3],keep_dims=False)
        squeeze = tf.expand_dims(squeeze, axis=1)
        attn = tf.layers.Conv1D(filters=1,
            kernel_size=k_size,
            padding='same',
            kernel_initializer=conv_kernel_initializer(),
            use_bias=False,
            data_format=self._data_format)(squeeze)

        attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)
        attn = tf.math.sigmoid(attn)
        scale = x * attn
        return x * attn
