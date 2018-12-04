from model import ImageCaptionModel
from config import get_hparams
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import joblib
import time
import glob
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def get_num_records(files):
    compression = tf.python_io.TFRecordCompressionType.GZIP
    count = 0
    for fn in tqdm(files):
        for record in tf.python_io.tf_record_iterator(fn, options=tf.python_io.TFRecordOptions(compression)):
            count += 1
    return count

def train(training_filenames, num_train_records, model):
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(model.hps.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:                
            saver.restore(sess, ckpt.model_checkpoint_path)
            gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(model.global_step, gs))
        else:
            sess.run(tf.global_variables_initializer())

        training_handle = sess.run(model.training_iterator.string_handle())
        sess.run(model.training_iterator.initializer,
                 feed_dict={model.filenames: training_filenames})

        num_batch_per_epoch_train = num_train_records // model.hps.batch_size

        for epoch in range(model.hps.training_epochs):
            start = time.perf_counter()
            gs = model.global_step.eval()
            _loss = []
            for i in range(num_batch_per_epoch_train):
                train_loss_batch, _ = sess.run(
                        [model.total_loss, model.train_op],
                        feed_dict={model.handle: training_handle, model.max_len: [gs + 1]})
                _loss.append(train_loss_batch)
                if (i % 1000 == 0):
                  print("step: {0:d}, minibatch training loss: {1:.4f}".format(i, train_loss_batch))

            loss_this_epoch = np.sum(_loss)
            end = time.perf_counter()
            print('Epoch {:2d} - train loss: {:.4f} - time: {:4f}'.format(
                    epoch, np.mean(_loss), end - start))
            saver.save(sess, model.hps.ckpt_dir + 'model.ckpt', global_step=gs)
            print("save checkpoint in {}".format(model.hps.ckpt_dir + 'model.ckpt-' + str(gs)))  

if __name__ == '__main__':

    hparams = get_hparams()

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    model = ImageCaptionModel(hparams, mode='train')
    model.build()
    writer = tf.summary.FileWriter("../tensorboard/", graph = sess.graph)
    sess.close()

    training_filenames = glob.glob('../data/tfrecord/train-*')
    num_train_records = get_num_records(training_filenames)
    train(training_filenames, num_train_records, model)
