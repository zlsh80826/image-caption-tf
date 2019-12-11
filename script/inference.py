from model import ImageCaptionModel
from config import get_hparams
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import subprocess
import itertools
import joblib
import pickle
import os 

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
    candidates = list()

    start_word_feed = [st for _ in range(len(imgs_embed))]

    initial_states = sess.run(
            fetches=model.initial_state,
            feed_dict={'features:0': imgs_embed})

    top_probs, top_words, states = sess.run(
        fetches=['optimize/prob:0', 'optimize/prob:1', 'rnn_scope/final_state:0'],
        feed_dict={
            'input_feed:0': start_word_feed,
            'rnn_scope/state_feed:0': initial_states
        })

    for idx, (probs, words) in enumerate(zip(top_probs, top_words)):
        candidate = list()
        for p, w in zip(probs, words):
            candidate.append((p, [w], (states[0][0][idx][:], states[0][1][idx][:])))
        candidates.append(candidate)

    for i in tqdm(range(model.hps.max_caption_len - 1), desc='Beam search'):
        candidate = [[] for _ in range(len(imgs_embed))]
        for n in range(len(candidates[0])):
            feed_word = [s[n][1][-1] for s in candidates]
            feed_state = [[[s[n][2][0] for s in candidates], [s[n][2][1] for s in candidates]]]
            top_probs, top_words, states = sess.run(
                fetches=['optimize/prob:0', 'optimize/prob:1',
                         'rnn_scope/final_state:0'],
                feed_dict={
                    'input_feed:0': feed_word,
                    'rnn_scope/state_feed:0': feed_state
                })
            for idx, (probs, words) in enumerate(zip(top_probs, top_words)):
                for p, w in zip(probs, words):
                    if candidates[idx][n][1][-1] is ed:
                        candidate[idx].append((candidates[idx][n][0], candidates[idx][n][1] + [ed],
                            (states[0][0][idx][:], states[0][1][idx][:])))
                    else:
                        candidate[idx].append((candidates[idx][n][0] * p, candidates[idx][n][1] + [w],
                            (states[0][0][idx][:], states[0][1][idx][:])))
        candidates = list()
        for c in candidate:
            candidates.append(sorted(c, reverse=True)[:model.hps.queue_size])

    caption_ids = [c[0][1] for c in candidates]
    probs = [c[0][1] for c in candidates]

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
    encode_map = pickle.load(open('../data/enc_map.pkl', 'rb'))  # token => id
    decode_map = pickle.load(open('../data/dec_map.pkl', 'rb'))  # id => token
    return encode_map, decode_map

if __name__ == '__main__':
    
    encode_map, decode_map = read_map()
    img_test = joblib.load('../data/processed/valid.joblib')
    hps = get_hparams()

    tf.reset_default_graph()
    model = ImageCaptionModel(hps, mode='test')
    model.build()
    
    predict = generate_captions(model, decode_map=decode_map, encode_map=encode_map, img_test=img_test)
    predict.to_csv('../evaluation/captions.csv')
    os.chdir('../evaluation')
    subprocess.run(['./gen_score', '-i', 'captions.csv', '-r', 'score.csv'])
