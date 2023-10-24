import tensorflow as tf
from .crflayer import CRFLayer
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Embedding,
    Dense,
    TimeDistributed,
    Dropout,
    Bidirectional
)
from tensorflow.keras import backend as K


class NerModel(tf.keras.Model):
    '''
    NerModel is a Named Entity Recognition (NER) model that uses a Bidirectional LSTM with CRF layer for sequence tagging.

    Args:
        hidden_num (int): Number of hidden units in the LSTM layer.
        vocab_size (int): Size of the vocabulary.
        label_size (int): Number of NER labels.
        embedding_size (int): Size of the word embeddings.
        name (str, optional): Name of the model. Default is 'BilstmCrfModel'.

    Methods:
        call(text: tf.Tensor, labels=None, training=None) -> tf.Tensor: Perform forward pass for the NER model.

    '''

    def __init__(
        self,
        hidden_num: int,
        vocab_size: int,
        label_size: int,
        embedding_size: int,
        name: str = 'BilstmCrfModel',
        **kwargs
    ) -> None:
        super(NerModel, self).__init__(name=name, **kwargs)
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding = Embedding(
            vocab_size,
            embedding_size,
            mask_zero=True,
            name='embedding'
        )

        self.biLSTM = Bidirectional(
            LSTM(
                hidden_num,
                return_sequences=True
            ),
            name='bilstm'
        )

        self.dense = TimeDistributed(
            Dense(label_size), name='dense'
        )

        self.crf = CRFLayer(label_size, name='crf')

    def call(self, text: tf.Tensor, labels=None, training=None) -> tf.Tensor:
        '''
        Perform forward pass for the NER model.

        Args:
            text (tf.Tensor): Input text data.
            labels (tf.Tensor, optional): True labels. Default is None.
            training (bool, optional): Flag indicating if the model is in training mode. Default is None.

        Returns:
            tf.Tensor: Predicted NER labels.
        '''
        seq_lengths = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(text, 0), tf.int32),
            axis=-1
        )

        if training is None:
            training = K.learning_phase()

        inputs = self.embedding(text)
        bilstm = self.biLSTM(inputs)
        logits = self.dense(bilstm)
        outputs = self.crf(logits, seq_lengths, training)

        return outputs
