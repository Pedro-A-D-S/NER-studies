import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class CRFLayer(Layer):
    '''
    Computes the log likelihood during training
    Performs Viterbi decoding during prediction.
    '''

    def __init__(
        self,
        label_size,
        mask_id: int = 0,
        trans_params=None,
        name: str = 'crf',
        **kwargs
    ):
        super(CRFLayer, self).__init__(name=name, **kwargs)
        self.label_size = label_size
        self.mask_id = mask_id
        self.transition_params = None

        if trans_params is None:
            self.transition_params = tf.Variable(
                tf.random.uniform(shape=(label_size, label_size)),
                trainable=False
            )

        else:
            self.transition_params = trans_params
