import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm

processed_dir = ['train2014', 'valid2014']

def read_test_ids(test_file):
    df = pd.read_csv(test_file, header=0)
    ids = df['img_id']
    return [int(id_[:-4]) for id_ in ids]

def create_new_data_dir():
    os.makedirs('train', exist_ok=True)
    os.makedirs('valid', exist_ok=True)

def process_dir(target_dir, test_ids):
    files = glob.glob(target_dir + '/*.jpg')
    for f in tqdm(files):
        id_ = int(f.split('_')[2][:-4])
        if id_ in test_ids:
            shutil.move(f, 'valid/' + str(id_) + '.jpg')
        else:
            shutil.move(f, 'train/' + str(id_) + '.jpg')
 
if __name__ == '__main__':

    create_new_data_dir()
    test_ids = read_test_ids('test.csv')
    process_dir('val2014', test_ids)
    process_dir('train2014', test_ids)

    os.redir('train2014')
    os.redir('val2014')
