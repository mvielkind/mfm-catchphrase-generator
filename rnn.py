import os
import json
import numpy as np
import argparse
import boto3
import tensorflow as tf


def process_data(batch_size):
    """
    Creates the input data from the transcript file for the RNN.
    :param batch_size: int. Size of the batches to process.
    :return: Dataset to train on and raw vocabulary of characters in the transcript.
    """
    s3 = boto3.client('s3')

    raw = s3.get_object(Bucket='mfm-bot-recordings', Key='main_transcript.txt')
    text = raw["Body"].read()

    # The unique characters in the file to map characters to integers.
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}

    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for a single input in characters
    seq_length = 500

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Batch / buffer size to shuffle dataset.
    BATCH_SIZE = batch_size
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset, vocab


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Build the TensorFlow model using the parameters provided by the user.
    :return: TensorFlow model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--embed-dim', type=int)
    parser.add_argument('--rnn-units', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)

    args, _ = parser.parse_known_args()

    embedding_dim = args.embed_dim
    rnn_units = args.rnn_units
    batch_size = args.batch_size
    epochs = args.epochs
    dataset, vocab = process_data(batch_size, )

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)

    model.compile(optimizer='adam', loss=loss)

    # Path to save checkpoints to.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        monitor='loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True)

    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

    # Build and save the final model based on the best model from training.
    model = build_model(len(vocab), embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.save('/opt/ml/model/000000001', save_format='tf')
