from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import keras
import tensorflow.compat.v1 as tf
import sys

sys.path.append('../')
from holstep_training import unconditioned_classification_models
from holstep_training import conditioned_classification_models
from holstep_training import data_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_name',
                           'cnn_2x',
                           'Name of model to train.')
tf.app.flags.DEFINE_string('task_name',
                           'unconditioned_classification',
                           'Name of task to run: "conditioned_classification" '
                           'or "unconditioned_classification".')

training_batch_size = 64
training_max_len = 512


def main(_):
    logging.basicConfig(level=logging.DEBUG)
    if not os.path.exists('/tmp/hol'):
        os.makedirs('/tmp/hol')


    # Parse the training and validation data.
    parser = data_utils.DataParser('./holstep', use_tokens=False,
                                   verbose=1)

    # Print useful stats about the parsed data.
    logging.info('Training data stats:')
    parser.display_stats(parser.train_conjectures)
    logging.info('---')
    logging.info('Validation data stats:')
    parser.display_stats(parser.val_conjectures)

    voc_size = len(parser.vocabulary) + 1

    if FLAGS.task_name == 'conditioned_classification':
        # Get the function for building the model, and the encoding to use.
        make_model, encoding = conditioned_classification_models.MODELS.get(
            FLAGS.model_name, None)
        if not make_model:
            raise ValueError('Unknown model:', FLAGS.model_name)

        # Instantiate a generator that will yield batches of training data.
        train_generator = parser.training_steps_and_conjectures_generator(
            encoding=encoding, max_len=training_max_len,
            batch_size=training_batch_size)

        # Instantiate a generator that will yield batches of validation data.
        val_generator = parser.validation_steps_and_conjectures_generator(
            encoding=encoding, max_len=training_max_len,
            batch_size=training_batch_size)

    elif FLAGS.task_name == 'unconditioned_classification':
        make_model, encoding = unconditioned_classification_models.MODELS.get(
            FLAGS.model_name, None)
        if not make_model:
            raise ValueError('Unknown model:', FLAGS.model_name)

        train_generator = parser.training_steps_generator(
            encoding=encoding, max_len=training_max_len,
            batch_size=training_batch_size)

        val_generator = parser.validation_steps_generator(
            encoding=encoding, max_len=training_max_len,
            batch_size=training_batch_size)

    else:
        raise ValueError('Unknown task_name:', FLAGS.task_name)

    
    model = make_model(voc_size, training_max_len)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['acc'])

    # Define a callback for saving the model to the log directory.
    checkpoint_path = os.path.join('/tmp/hol', FLAGS.model_name + '.h5')
    checkpointer = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True)

    # Define a callback for writing TensorBoard logs to the log directory.
    tensorboard_vis = keras.callbacks.TensorBoard(log_dir='/tmp/hol')

    logging.info('Fit model...')
    history = model.fit(train_generator, epochs=5, steps_per_epoch=10, validation_data=val_generator,
                        validation_steps=100)

    # Save training history to a JSON file.
    f = open(os.path.join('/tmp/hol', 'history.json'), 'w')
    f.write(json.dumps(history.history))
    f.close()

if __name__ == '__main__':
    tf.app.run()