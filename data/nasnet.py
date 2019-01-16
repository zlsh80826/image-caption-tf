import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.applications.nasnet import NASNetLarge, preprocess_input, decode_predictions
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import warnings
import joblib
import glob
import time
import os
warnings.filterwarnings('ignore')


def preprocess_image(filename):
    img = image.load_img(filename, target_size=(331, 331, 3))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

def get_files(version):
    files = glob.glob(version + '/*')
    return files

def predict(version, model):
    batch_size = 2048
    files = get_files(version)
    num_batch = len(files) // batch_size + 1
    results = dict()
    for idx in tqdm(range(num_batch)):
        start = idx * batch_size
        # end = min(len(files), (idx + 1) * batch_size)
        end = (idx + 1) * batch_size
        batch = files[start:end]
        batch_images = list()
        for f in batch:
            batch_images.append(preprocess_image(f))
        batch_results = model.predict(np.asarray(batch_images))
        for filename, result in zip(batch, batch_results):
            results[os.path.basename(filename)] = result
    return results

if __name__ == '__main__':

    model = NASNetLarge(input_shape=(331, 331, 3), include_top=False, pooling='max')
    os.makedirs('processed', exist_ok=True)

    results = predict('valid', model)
    joblib.dump(results, 'processed/valid.joblib', compress=3)

    results = predict('train', model)
    num_train = len(results)
    num_partition = 10
    psize = num_train // num_partition + 1
    key_list = list(results)

    for p in range(num_partition):
        start = p * psize
        end = (p + 1) * psize
        part_results = dict()        
        for key in key_list[start:end]:
            part_results[key] = results[key]

        joblib.dump(part_results, 'processed/train.{}.joblib'.format(p), compress=3)
