from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
from time import time

import tensorflow.compat.v1 as tf
from keras import layers
from keras.models import Model

sys.path.append('../')
from holstep_training import data_utils

training_batch_size = 64
training_max_len = 512
epochs = 5
steps_per_epoch = 10
validation_steps = 10

def cnn_2x(voc_size, max_len, dropout=0.5):
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    embedded_statement = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation='relu')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model

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

    make_model = cnn_2x
    encoding = 'integer'

    train_generator = parser.training_steps_generator(
        encoding=encoding, max_len=training_max_len,
        batch_size=training_batch_size)

    val_generator = parser.validation_steps_generator(
        encoding=encoding, max_len=training_max_len,
        batch_size=training_batch_size)

    
    model = make_model(voc_size, training_max_len)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                    loss='binary_crossentropy',
                    metrics=['acc'])

    print("Training Started:")
    model.fit( train_generator, epochs=epochs, 
                                steps_per_epoch=steps_per_epoch, 
                                validation_data=val_generator,
                                validation_steps=validation_steps )

    t = time.time()
    export_path = "./{}.h5".format(int(t))
    model.save(export_path)

if __name__ == '__main__':
    tf.app.run()
