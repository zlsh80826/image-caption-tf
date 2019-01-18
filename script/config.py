import tensorflow as tf
import pickle

feature_dim = 4032
decode_map = pickle.load(open('../data/dec_map.pkl', 'rb'))
vocab_dim = len(decode_map)

def get_hparams():
    hparams = tf.contrib.training.HParams(
        feature_dim=feature_dim,
        image_embedding_dim=1536,
        word_embedding_dim=512,
        vocab_dim=vocab_dim,
        training_epochs=20,
        max_caption_len=16, 
        hidden_dim=1024,
        batch_size=64,
        num_layers=1,
        queue_size=3,
        beam_size=3,
        dropout=0.5,
        lr=1e-4,
        ckpt_dir='../model/')
    return hparams
