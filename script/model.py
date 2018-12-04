import tensorflow as tf
from config import feature_dim

def parse_records(record):
    
    features = {
      "img": tf.FixedLenFeature([feature_dim], dtype=tf.float32),
      "caption": tf.VarLenFeature(dtype=tf.int64)
    }

    features = tf.parse_single_example(record, features=features)

    img = features['img']
    caption = tf.cast(features['caption'].values, tf.int32)

    # create input and output sequence for each training example
    # e.g. caption :   [0 2 5 7 9 1]
    #      input_seq:  [0 2 5 7 9]
    #      output_seq: [2 5 7 9 1]
    #      mask:       [1 1 1 1 1]
    caption_len = tf.shape(caption)[0]
    input_len = tf.expand_dims(tf.subtract(caption_len, 1), 0)

    input_seq = tf.slice(caption, [0], input_len)
    output_seq = tf.slice(caption, [1], input_len)
    mask = tf.ones(input_len, dtype=tf.int32)

    records = {
      'img': img,
      'input_seq': input_seq,
      'output_seq': output_seq,
      'mask': mask
    }

    return records

def tfrecord_iterator(filenames, batch_size, record_parser):

    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP", num_parallel_reads=8)
    dataset = dataset.map(record_parser, num_parallel_calls=8)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={
            'img': [None],
            'input_seq': [None],
            'output_seq': [None],
            'mask': [None]
        },
        padding_values={
            'img': 1.,        # needless, for completeness
            'input_seq': 1,   # padding input sequence in this batch
            'output_seq': 1,  # padding output sequence in this batch
            'mask': 0         # padding 0 means no words in this position
        }
    )  

    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size)

    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes

    return iterator, output_types, output_shapes

class ImageCaptionModel(object):

    def __init__(self, hparams, mode):
        self.hps = hparams
        self.mode = mode
        self.regularizer = tf.contrib.layers.l2_regularizer(1e-3)
        
    def _build_inputs(self):
        if self.mode == 'train':
            self.filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
            self.training_iterator, types, shapes = tfrecord_iterator(
                    self.filenames, self.hps.batch_size, parse_records)

            self.handle = tf.placeholder(tf.string, shape=[], name='handle')
            self.max_len = tf.placeholder(tf.int32, shape=[1], name='max_len')
            iterator = tf.data.Iterator.from_string_handle(self.handle, types, shapes)
            records = iterator.get_next()

            features = records['img']
            features.set_shape([None, self.hps.feature_dim])
            captions_in = records['input_seq']
            captions_out = records['output_seq']
            input_mask = records['mask']

        else:
            features = tf.placeholder(
                    tf.float32,
                    shape=[None, self.hps.feature_dim],
                    name='features')
            input_feed = tf.placeholder(tf.int32, shape=[None], name='input_feed')
            captions_in = tf.expand_dims(input_feed, axis=1)
            captions_out = None
            input_mask = None

        self.features = features
        self.captions_in = captions_in
        self.captions_out = captions_out
        self.input_mask = input_mask

    
    def _build_seq_embeddings(self):
        with tf.variable_scope('seq_embedding'), tf.device('/gpu:0'):
            embedding_matrix = tf.get_variable(
                name='embedding_matrix',
                shape=[self.hps.vocab_dim, self.hps.word_embedding_dim], regularizer=self.regularizer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.captions_in)

        self.seq_embeddings = seq_embeddings
        
    def _batch_norm(self, x):
        return tf.contrib.layers.batch_norm(inputs=x, decay=0.95, center=True,
                                            scale=True, is_training=(self.mode=='train'),
                                            updates_collections=None)
    
    def _get_initial_state(self, features):
        with tf.variable_scope('initial_state'):
            c = tf.layers.dense(features, self.hps.image_embedding_dim, activation=tf.nn.selu, 
                                kernel_regularizer=self.regularizer, name='proj_c')
            h = tf.layers.dense(features, self.hps.image_embedding_dim, activation=tf.nn.selu, 
                                kernel_regularizer=self.regularizer, name='proj_h')
            
            if self.mode == 'train':
                c = tf.nn.dropout(c, keep_prob=self.hps.dropout)
                h = tf.nn.dropout(h, keep_prob=self.hps.dropout)
                
            return c, h
                
    def _get_rnn_cell(self):
        with tf.variable_scope('cell'):
            rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=self.hps.hidden_dim, state_is_tuple=True)
                        for i in range(self.hps.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)
            if self.mode == 'train':
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                    rnn_cell,
                    input_keep_prob=0.5,
                    output_keep_prob=0.5)
            return rnn_cell

    def _build_model(self):

        features = self._batch_norm(self.features)
        c, h = self._get_initial_state(features)
        rnn_cell = self._get_rnn_cell()
        initial_state = (tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h),)
        
        with tf.variable_scope('rnn_scope/', initializer=tf.contrib.layers.xavier_initializer(), 
                reuse=tf.AUTO_REUSE) as rnn_scope:            
            self.initial_state = initial_state

            if self.mode == 'train':
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                max_len = tf.broadcast_to(self.max_len, [tf.shape(self.features)[0]])
                length = tf.minimum(sequence_length, max_len)
                
                outputs, _ = tf.nn.dynamic_rnn(
                    cell=rnn_cell, inputs=self.seq_embeddings,
                    sequence_length=length, initial_state=initial_state,
                    parallel_iterations=2048, dtype=tf.float32, scope=rnn_scope)
                                            
            else:
                state_feed = tf.placeholder(tf.float32, shape=[self.hps.num_layers, 2, None, self.hps.hidden_dim], name='state_feed') 
                unstack_state = tf.unstack(state_feed, axis=0)
                tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(self.hps.num_layers)])
                outputs, state = rnn_cell(inputs=tf.squeeze(self.seq_embeddings, axis=[1]), state=tuple_state)
                final_state = tf.convert_to_tensor(state, dtype=tf.float32, name='final_state')

        outputs = self._batch_norm(outputs)
        rnn_outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])
            
        with tf.variable_scope("logits") as logits_scope:
            rnn_outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])
            logits = tf.layers.dense(rnn_outputs, self.hps.vocab_dim, kernel_regularizer=self.regularizer)

        with tf.name_scope('optimize') as optimize_scope:
            if self.mode == 'train':
                targets = tf.reshape(self.captions_out, [-1])
                indicator = tf.cast(tf.reshape(self.input_mask, [-1]), tf.float32)

                # loss function
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                batch_loss = tf.div(
                        tf.reduce_sum(tf.multiply(losses, indicator)),
                        tf.reduce_sum(indicator),
                        name='batch_loss')

                self.total_loss = batch_loss # + tf.losses.get_regularization_loss()

                # save checkpoint
                self.global_step = tf.train.get_or_create_global_step()

                optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
                grads_and_vars = optimizer.compute_gradients(self.total_loss, tf.trainable_variables())
                clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 0.05), gv[1]) for gv in grads_and_vars]
                self.train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

            else:
                pred_softmax = tf.nn.softmax(logits, name='softmax')
                prediction = tf.argmax(pred_softmax, axis=1, name='prediction')

                
    def build(self):
        self._build_inputs()
        self._build_seq_embeddings()
        self._build_model()
