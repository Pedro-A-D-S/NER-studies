import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class CRFLayer(Layer):
    '''
    CRFLayer computes the log likelihood during training and performs Viterbi decoding during prediction.

    Args:
        label_size (int): The number of labels in the CRF layer.
        mask_id (int, optional): The mask id used in input sequences. Default is 0.
        trans_params (tf.Tensor, optional): Pre-defined transition parameters. Default is None, which initializes random parameters.
        name (str, optional): Name of the layer. Default is 'crf'.

    Attributes:
        label_size (int): The number of labels in the CRF layer.
        mask_id (int): The mask id used in input sequences.
        transition_params (tf.Tensor): Transition parameters for the CRF layer.

    Methods:
        get_seq_lengths(matrix: tf.Tensor) -> tf.Tensor: Get the sequence lengths based on the input matrix.
        call(inputs: tf.Tensor, seq_lengths: tf.Tensor, training=None) -> tf.Tensor: Perform CRF computation during training or Viterbi decoding during prediction.
        pad_viterbi(viterbi: list, max_seq_len: int) -> list: Pad the Viterbi path to match the maximum sequence length.
        get_proper_labels(y_true: tf.Tensor) -> tf.Tensor: Get the proper labels from the true labels.
        loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: Compute the CRF loss during training.

    '''

    def __init__(
        self,
        label_size: int,
        mask_id: int = 0,
        trans_params: tf.Tensor = None,
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

    def get_seq_lengths(self, matrix: tf.Tensor) -> tf.Tensor:
        '''
        Get the sequence lengths based on the input matrix.

        Args:
            matrix (tf.Tensor): Input matrix with mask values.

        Returns:
            tf.Tensor: Sequence lengths as a 1-D tensor.
        '''
        mask = tf.not_equal(matrix, self.mask_id)
        seq_lengths = tf.math.reduce_sum(
            tf.cast(mask, dtype=tf.int32),
            axis=-1
        )
        return seq_lengths

    def call(self, inputs: tf.Tensor, seq_lengths: tf.Tensor, training=None) -> tf.Tensor:
        '''
        Perform CRF computation during training or Viterbi decoding during prediction.

        Args:
            inputs (tf.Tensor): Input sequence data.
            seq_lengths (tf.Tensor): Sequence lengths.
            training (bool, optional): Flag indicating if the layer is in training mode. Default is None.

        Returns:
            tf.Tensor: CRF output or Viterbi decoding results.
        '''
        if training is None:
            training = K.learning_phase()

        if training:
            return inputs

        _, max_seq_len, _ = inputs.shape
        seqlens = seq_lengths
        paths = []

        for logit, text_len in zip(inputs, seqlens):
            viterbi_path, _ = tfa.text.viterbi_decode(
                logit[:text_len],
                self.transition_params
            )
            paths.append(self.pad_viterbi(viterbi_path, max_seq_len))

        return tf.convert_to_tensor(paths)

    def pad_viterbi(self, viterbi: list, max_seq_len: int) -> list:
        '''
        Pad the Viterbi path to match the maximum sequence length.

        Args:
            viterbi (list): Viterbi path as a list.
            max_seq_len (int): Maximum sequence length to pad to.

        Returns:
            list: Padded Viterbi path.
        '''
        if len(viterbi) < max_seq_len:
            viterbi = viterbi + [self.mask_id] * (max_seq_len - len(viterbi))
        return viterbi

    def get_proper_labels(self, y_true: tf.Tensor) -> tf.Tensor:
        '''
        Get the proper labels from the true labels.

        Args:
            y_true (tf.Tensor): True labels.

        Returns:
            tf.Tensor: Proper labels.
        '''
        shape = y_true.shape
        if len(shape) > 2:
            return tf.argmax(y_true, -1, output_type=tf.int32)
        return y_true

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        '''
        Compute the CRF loss during training.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Predicted labels.

        Returns:
            tf.Tensor: CRF loss.
        '''
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(self.get_proper_labels(y_true), y_pred.dtype)

        seq_lengths = self.get_seq_lengths(y_true)
        log_likelihoods, self.transition_params = tfa.text.crf_log_likelihood(
            y_pred,
            y_true,
            seq_lengths
        )

        self.transition_params = tf.Variable(
            self.transition_params, trainable=False)
        loss = -tf.reduce_mean(log_likelihoods)

        return loss
