import tensorflow as tf
import multiprocessing
from tqdm import tqdm
import pandas as pd
import joblib
import glob
import os 

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tfrecords(cap, img, output):

    compression = tf.python_io.TFRecordCompressionType.GZIP
    writer = tf.python_io.TFRecordWriter(output, options=tf.python_io.TFRecordOptions(compression))
    with tqdm(total=len(img)) as pbar:
        for idx, row in img.iterrows():
            image = row['img']

            for _, inner_row in cap[cap['img_id'] == row['img_id']].iterrows():
                caption = eval(inner_row['caption'])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': _float_feature(image),
                    'caption': _int64_feature(caption)
                }))
                writer.write(example.SerializeToString())
            pbar.update(1)
    writer.close()
        
caption = pd.read_csv('../data/train_enc_cap.csv') 

def task(i):
    img_train = joblib.load('../data/processed/train.{}.joblib'.format(i))
    img_train_df = pd.DataFrame(list(img_train.items()), columns=['img_id', 'img'])
    create_tfrecords(caption, img_train_df, '../data/tfrecords/train.{}.tfrecord'.format(i + 1))

if __name__ == '__main__':

    os.makedirs('../data/tfrecords', exist_ok=True)

    num_joblib = len(glob.glob('../data/processed/train.*.joblib'))
    print('Number of training files:', num_joblib)

    with multiprocessing.Pool(num_joblib) as p:
        for i, _ in enumerate(p.imap_unordered(task, range(num_joblib))):
            print(i)


