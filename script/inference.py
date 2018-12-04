from model import ImageCaptionModel
from config import get_hparams
import tensorflow as tf
import pandas as pd
import subprocess
import itertools
import joblib
import pickle
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def inference(model, img_embed, encode_map, decode_map, batch_size):
    saver = tf.train.Saver()        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(model.hps.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:                
            saver.restore(sess, tf.train.latest_checkpoint(model.hps.ckpt_dir))
            caption = predict(model, sess, img_embed, encode_map, decode_map, batch_size)
            return caption
        else:
            print("No checkpoint found.")

def predict(model, sess, imgs_embed, encode_map, decode_map, batch_size):
    st, ed = encode_map['<ST>'], encode_map['<ED>']

    caption_id = list()
    start_word_feed = [st for _ in range(len(imgs_embed))]

    initial_states = sess.run(
            fetches=model.initial_state,
            feed_dict={'features:0': imgs_embed})

    next_words, states = sess.run(
        fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
        feed_dict={
            'input_feed:0': start_word_feed,
            'rnn_scope/state_feed:0': initial_states
        })
    caption_id.append([int(word) for word in next_words])

    for i in range(model.hps.max_caption_len - 1):
        next_words, states = sess.run(
            fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
            feed_dict={
                'input_feed:0': next_words,
                'rnn_scope/state_feed:0': states
            })
        caption_id.append([int(word) for word in next_words])

    caption_ids = list(map(list, zip(*caption_id)))

    captions = [
        [decode_map[x] for x in caption_id[:None if ed not in caption_id else caption_id.index(ed)]]
        for caption_id in caption_ids
    ]

    return [' '.join(caption) for caption in captions]

def generate_captions(model, encode_map, decode_map, img_test, max_len=16, batch_size=512):
    img_ids, caps = list(), list()
    
    img_test = list(img_test.items())
    num_batches = len(img_test) // batch_size

    batches = []
    cnt = 0
    batch = []
    for t, (img_id, img) in enumerate(img_test):
        if cnt is batch_size:
            batches.append(batch)
            cnt = 0
            batch = []
        batch.append((img_id, img))
        cnt += 1
    batches.append(batch)
    
    for t, batch in enumerate(batches):
        img_ids.append([img_id for (img_id, img) in batch])
        imgs = [img for (img_id, img) in batch]
        caps.append(inference(model, imgs, encode_map, decode_map, batch_size))
        t += 1
    
    img_ids = list(itertools.chain.from_iterable(img_ids))
    caps = list(itertools.chain.from_iterable(caps))
    
    return pd.DataFrame({
        'img_id': img_ids,
        'caption': caps
    }).set_index(['img_id'])

def read_map():
    encode_map = pickle.load(open('../data/captions/enc_map.pkl', 'rb'))  # token => id
    decode_map = pickle.load(open('../data/captions/dec_map.pkl', 'rb'))  # id => token
    return encode_map, decode_map

if __name__ == '__main__':
    
    encode_map, decode_map = read_map()
    img_test = joblib.load('../data/test.joblib')
    hparams = get_hparams()

    tf.reset_default_graph()
    model = ImageCaptionModel(hparams, mode='test')
    model.build()
    
    predict = generate_captions(model, decode_map=decode_map, encode_map=encode_map, img_test=img_test)
    predict.to_csv('../output/captions.csv')
    subprocess.run(['sh', 'gen_score.sh'])
