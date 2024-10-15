"""
The trained 1900-dimensional mLSTM babbler.
Source: https://github.com/churchlab/UniRep/blob/master/unirep.py
"""

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_addons as tfa
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.unirep.utils import aa_seq_to_int, int_to_aa
from baselines.unirep.utils.unirep_utils import bucketbatchpad

# Helpers
def tf_get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims

def sample_with_temp(logits, t):
    """
    Takes temperature between 0 and 1 -> zero most conservative, 1 most liberal. Samples.
    """
    t_adjusted = logits / t  # broadcast temperature normalization
    softed = tf.nn.softmax(t_adjusted)
    
    # Make a categorical distribution from the softmax and sample
    return tfp.distributions.Categorical(probs=softed).sample()

def initialize_uninitialized(sess):
    """
    from https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
    """
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


# Setup to initialize from the correctly named model files.
class mLSTMCell1900(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 model_path="./",
                 wn=True,
                 scope='mlstm',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCell1900, self).__init__()
        self._num_units = num_units
        self._model_path = model_path
        self._wn = wn
        self._scope = scope
        self._var_device = var_device

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        nin = inputs.get_shape()[1].value

        # Unpack the state tuple
        c_prev, h_prev = state
        with tf.variable_scope(self._scope):
            wx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wx:0.npy"))
            wh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wh:0.npy"))
            wmx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmx:0.npy"))
            wmh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_wmh:0.npy"))
            b_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_b:0.npy"))
            gx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gx:0.npy"))
            gh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gh:0.npy"))
            gmx_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmx:0.npy"))
            gmh_init = np.load(os.path.join(self._model_path, "rnn_mlstm_mlstm_gmh:0.npy"))        
            wx = tf.get_variable(
                "wx", initializer=wx_init)
            wh = tf.get_variable(
                "wh", initializer=wh_init)
            wmx = tf.get_variable(
                "wmx", initializer=wmx_init)
            wmh = tf.get_variable(
                "wmh", initializer=wmh_init)
            b = tf.get_variable(
                "b", initializer=b_init)
            if self._wn:
                gx = tf.get_variable(
                    "gx", initializer=gx_init)
                gh = tf.get_variable(
                    "gh", initializer=gh_init)
                gmx = tf.get_variable(
                    "gmx", initializer=gmx_init)
                gmh = tf.get_variable(
                    "gmh", initializer=gmh_init)

        if self._wn:
            wx = tf.nn.l2_normalize(wx, axis=0) * gx
            wh = tf.nn.l2_normalize(wh, axis=0) * gh
            wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
            wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh
        m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c_prev + i * u
        h = o * tf.tanh(c)
        return h, (c, h)

class mLSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 wx_init=tf.orthogonal_initializer(),
                 wh_init=tf.orthogonal_initializer(),
                 wmx_init=tf.orthogonal_initializer(),
                 wmh_init=tf.orthogonal_initializer(),
                 b_init=tf.orthogonal_initializer(),
                 gx_init=tf.ones_initializer(),
                 gh_init=tf.ones_initializer(),
                 gmx_init=tf.ones_initializer(),
                 gmh_init=tf.ones_initializer(),
                 wn=True,
                 scope='mlstm',
                 var_device='cpu:0',
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCell, self).__init__()
        self._num_units = num_units
        self._wn = wn
        self._scope = scope
        self._var_device = var_device
        self._wx_init = wx_init
        self._wh_init = wh_init
        self._wmx_init = wmx_init
        self._wmh_init = wmh_init
        self._b_init = b_init
        self._gx_init = gx_init
        self._gh_init = gh_init
        self._gmx_init = gmx_init
        self._gmh_init = gmh_init

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10
        nin = inputs.get_shape()[1].value

        # Unpack the state tuple
        c_prev, h_prev = state
        with tf.variable_scope(self._scope):
            wx = tf.get_variable(
                "wx", initializer=self._wx_init)
            wh = tf.get_variable(
                "wh", initializer=self._wh_init)
            wmx = tf.get_variable(
                "wmx", initializer=self._wmx_init)
            wmh = tf.get_variable(
                "wmh", initializer=self._wmh_init)
            b = tf.get_variable(
                "b", initializer=self._b_init)
            if self._wn:
                gx = tf.get_variable(
                    "gx", initializer=self._gx_init)
                gh = tf.get_variable(
                    "gh", initializer=self._gh_init)
                gmx = tf.get_variable(
                    "gmx", initializer=self._gmx_init)
                gmh = tf.get_variable(
                    "gmh", initializer=self._gmh_init)

        if self._wn:
            wx = tf.nn.l2_normalize(wx, dim=0) * gx
            wh = tf.nn.l2_normalize(wh, dim=0) * gh
            wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
            wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh
        m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c_prev + i * u
        h = o * tf.tanh(c)
        return h, (c, h)

class mLSTMCellStackNPY(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units=256,
                 num_layers=4,
                 dropout=None,
                 res_connect=False,
                 wn=True,
                 scope='mlstm_stack',
                 var_device='cpu:0',
                 model_path="./"
                 ):
        # Really not sure if I should reuse here
        super(mLSTMCellStackNPY, self).__init__()
        self._model_path=model_path
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout = dropout
        self._res_connect = res_connect
        self._wn = wn
        self._scope = scope
        self._var_device = var_device
        bs = "rnn_mlstm_stack_mlstm_stack" # base scope see weight file names
        join = lambda x: os.path.join(self._model_path, x)
        layers = [mLSTMCell(
            num_units=self._num_units,
            wn=self._wn,
            scope=self._scope + str(i),
            var_device=self._var_device,
            wx_init=np.load(join(bs + "{0}_mlstm_stack{1}_wx:0.npy".format(i,i))),
            wh_init=np.load(join(bs + "{0}_mlstm_stack{1}_wh:0.npy".format(i,i))),
            wmx_init=np.load(join(bs + "{0}_mlstm_stack{1}_wmx:0.npy".format(i,i))),
            wmh_init=np.load(join(bs + "{0}_mlstm_stack{1}_wmh:0.npy".format(i,i))),
            b_init=np.load(join(bs + "{0}_mlstm_stack{1}_b:0.npy".format(i,i))),
            gx_init=np.load(join(bs + "{0}_mlstm_stack{1}_gx:0.npy".format(i,i))),
            gh_init=np.load(join(bs + "{0}_mlstm_stack{1}_gh:0.npy".format(i,i))),
            gmx_init=np.load(join(bs + "{0}_mlstm_stack{1}_gmx:0.npy".format(i,i))),
            gmh_init=np.load(join(bs + "{0}_mlstm_stack{1}_gmh:0.npy".format(i,i)))      
                 ) for i in range(self._num_layers)]
        if self._dropout:
            layers = [
                tf.nn.rnn_cell.DropoutWrapper(
                    layer, output_keep_prob=1-self._dropout) for layer in layers[:-1]] + layers[-1:]
        self._layers = layers

    @property
    def state_size(self):
        # The state is a tuple of c and h
        return (
            tuple(self._num_units for _ in range(self._num_layers)), 
            tuple(self._num_units for _ in range(self._num_layers))
            )

    @property
    def output_size(self):
        # The output is h
        return (self._num_units)

    def zero_state(self, batch_size, dtype):
        c_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
        h_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
        return (c_stack, h_stack)

    def call(self, inputs, state):
        # Inputs will be a [batch_size, input_dim] tensor.
        # Eg, input_dim for a 10-D embedding is 10

        # Unpack the state tuple
        c_prev, h_prev = state
        
        new_outputs = []
        new_cs = []
        new_hs = []
        for i, layer in enumerate(self._layers):
            if i == 0:
                h, (c,h_state) = layer(inputs, (c_prev[i],h_prev[i]))
            else:
                h, (c,h_state) = layer(new_outputs[-1], (c_prev[i],h_prev[i]))
            new_outputs.append(h)
            new_cs.append(c)
            new_hs.append(h_state)
        
        if self._res_connect:
            # Make sure number of layers does not affect the scale of the output
            scale_factor = tf.constant(1 / float(self._num_layers))
            final_output = tf.scalar_mul(scale_factor,tf.add_n(new_outputs))
        else:
            final_output = new_outputs[-1]

        return final_output, (tuple(new_cs), tuple(new_hs))

    
class babbler1900():

    def __init__(self,
                 model_path="./pbab_weights",
                 batch_size=256
                 ):
        self._rnn_size = 1900
        self._vocab_size = 26
        self._embed_dim = 10
        self._wn = True
        self._shuffle_buffer = 10000
        self._model_path = model_path
        self._batch_size = batch_size
        self._batch_size_placeholder = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self._minibatch_x_placeholder = tf.placeholder(
            tf.int32, shape=[None, None], name="minibatch_x")
        self._initial_state_placeholder = (
            tf.placeholder(tf.float32, shape=[None, self._rnn_size]),
            tf.placeholder(tf.float32, shape=[None, self._rnn_size])
        )
        self._minibatch_y_placeholder = tf.placeholder(
            tf.int32, shape=[None, None], name="minibatch_y")
        # Batch size dimensional placeholder which gives the
        # Lengths of the input sequence batch. Used to index into
        # The final_hidden output and select the stop codon -1
        # final hidden for the graph operation.
        self._seq_length_placeholder = tf.placeholder(
            tf.int32, shape=[None], name="seq_len")
        self._temp_placeholder = tf.placeholder(tf.float32, shape=[], name="temp")
        rnn = mLSTMCell1900(self._rnn_size, model_path=model_path, wn=self._wn)
        zero_state = rnn.zero_state(self._batch_size, tf.float32)
        single_zero = rnn.zero_state(1, tf.float32)
        mask = tf.sign(self._minibatch_y_placeholder)  # 1 for nonpad, zero for pad
        inverse_mask = 1 - mask  # 0 for nonpad, 1 for pad

        total_padded = tf.reduce_sum(inverse_mask)

        pad_adjusted_targets = (self._minibatch_y_placeholder - 1) + inverse_mask

        embed_matrix = tf.get_variable(
            "embed_matrix", dtype=tf.float32,
            initializer=np.load(os.path.join(self._model_path, "embed_matrix:0.npy"))
        )
        embed_cell = tf.nn.embedding_lookup(embed_matrix, self._minibatch_x_placeholder)
        self._output, self._final_state = tf.nn.dynamic_rnn(
            rnn,
            embed_cell,
            initial_state=self._initial_state_placeholder,
            swap_memory=True,
            parallel_iterations=1
        )
        
        # If we are training a model on top of the rep model, we need to access
        # the final_hidden rep from output. Recall we are padding these sequences
        # to max length, so the -1 position will not necessarily be the right rep.
        # to get the right rep, I will use the provided sequence length to index.
        # Subtract one for the last place
        indices = self._seq_length_placeholder - 1
        self._top_final_hidden = tf.gather_nd(self._output,
                tf.stack([tf.range(tf_get_shape(self._output)[0],
                    dtype=tf.int32), indices], axis=1))
        fmask = tf.cast(mask, tf.float32)[:, :, None]
        self._avg_hidden = tf.reduce_sum(fmask * self._output,
                axis=1) / tf.reduce_sum(fmask, axis=1)
        # LEFTOFF self._output is a batch size, seq_len, num_hidden.
        # I want to average along num_hidden, but I'll have to figure out how to mask out
        # the dimensions along sequence_length which are longer than the given sequence.
        flat = tf.reshape(self._output, [-1, self._rnn_size])
        if os.path.exists(os.path.join(self._model_path, "fully_connected_weights:0.npy")):
            weights_name="fully_connected_weights"
            bias_name="fully_connected_biases"
        else:
            weights_name="dense_kernel"
            bias_name="dense_bias"
        weights_init = tf.constant_initializer(
                np.load(os.path.join(self._model_path, f"{weights_name}:0.npy")))
        bias_init = tf.constant_initializer(
                np.load(os.path.join(self._model_path, f"{bias_name}:0.npy")))
        self.dense_layer = tf.keras.layers.Dense(self._vocab_size-1,
                activation=None, kernel_initializer=weights_init,
                bias_initializer=bias_init)
        logits_flat = self.dense_layer(flat)
        seqlen = tf_get_shape(self._minibatch_x_placeholder)[1]
        self._logits = tf.reshape(
            logits_flat, [batch_size, seqlen, self._vocab_size-1])
        self.batch_losses = tfa.seq2seq.sequence_loss(
            self._logits,
            tf.cast(pad_adjusted_targets, tf.int32),
            tf.cast(mask, tf.float32),
            average_across_batch=False
        )
        self._loss = tf.reduce_mean(self.batch_losses)
        self._sample = sample_with_temp(self._logits, self._temp_placeholder)
        with tf.Session() as sess:
            self._zero_state = sess.run(zero_state)
            self._single_zero = sess.run(single_zero)

     
    def get_rep(self,seq):
        """
        Input a valid amino acid sequence, 
        outputs a tuple of average hidden, final hidden, final cell representation arrays.
        Unfortunately, this method accepts one sequence at a time and is as such quite
        slow.
        """
        with tf.Session() as sess:
            initialize_uninitialized(sess)
            # Strip any whitespace and convert to integers with the correct coding
            int_seq = aa_seq_to_int(seq.strip())[:-1]
            # Final state is a cell_state, hidden_state tuple. Output is
            # all hidden states
            final_state_, hs = sess.run(
                [self._final_state, self._output], feed_dict={
                    self._batch_size_placeholder: 1,
                    self._minibatch_x_placeholder: [int_seq],
                    self._initial_state_placeholder: self._zero_state}
            )

        final_cell, final_hidden = final_state_
        # Drop the batch dimension so it is just seq len by
        # representation size
        final_cell = final_cell[0]
        final_hidden = final_hidden[0]
        hs = hs[0]
        avg_hidden = np.mean(hs, axis=0)
        return avg_hidden, final_hidden, final_cell

    def get_babble(self, seed, length=250, temp=1):
        """
        Return a babble at temperature temp (on (0,1] with 1 being the noisiest)
        starting with seed and continuing to length length.
        Unfortunately, this method accepts one sequence at a time and is as such quite
        slow.
        """
        with tf.Session() as sess:
            initialize_uninitialized(sess)
            int_seed = aa_seq_to_int(seed.strip())[:-1]
        
            # No need for padding because this is a single element
            seed_samples, final_state_ = sess.run(
                [self._sample, self._final_state], 
                feed_dict={
                    self._minibatch_x_placeholder: [int_seed],
                    self._initial_state_placeholder: self._zero_state, 
                    self._batch_size_placeholder: 1,
                    self._temp_placeholder: temp
                }
            )
            # Just the actual character prediction
            pred_int = seed_samples[0, -1] + 1
            seed = seed + int_to_aa[pred_int]
        
            for i in range(length - len(seed)):
                pred_int, final_state_ = sess.run(
                    [self._sample, self._final_state], 
                    feed_dict={
                        self._minibatch_x_placeholder: [[pred_int]],
                        self._initial_state_placeholder: final_state_, 
                        self._batch_size_placeholder: 1,
                        self._temp_placeholder: temp
                    }
                )
                pred_int = pred_int[0, 0] + 1
                seed = seed + int_to_aa[pred_int]
        return seed        
        
    def get_rep_ops(self):
        """
        Return tensorflow operations for the final_hidden state and placeholder.
        """
        return self._top_final_hidden, self._avg_hidden, self._minibatch_x_placeholder, self._batch_size_placeholder, self._seq_length_placeholder, self._initial_state_placeholder
        
    def get_babbler_ops(self):
        """
        Return tensorflow operations for 
        the logits, masked loss, minibatch_x placeholder, minibatch y placeholder, batch_size placeholder, initial_state placeholder
        Use if you plan on using babbler1900 as an initialization for another babbler, 
        eg for fine tuning the babbler to babble a differenct distribution.
        """
        return self._logits, self._loss, self._minibatch_x_placeholder, self._minibatch_y_placeholder, self._batch_size_placeholder, self._initial_state_placeholder

    def dump_weights(self,sess,dir_name="./1900_weights"):
        """
        Saves the weights of the model in dir_name in the format required 
        for loading in this module. Must be called within a tf.Session
        For which the weights are already initialized.
        """
        vs = tf.trainable_variables()
        for v in vs:
            name = v.name
            value = sess.run(v)
            np.save(os.path.join(dir_name,name.replace('/', '_') + ".npy"), np.array(value))
            


    def format_seq(self,seq,stop=False):
        """
        Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
        Here, the default is to strip the stop symbol (stop=False) which would have 
        otherwise been added to the end of the sequence. If you are trying to generate
        a rep, do not include the stop. It is probably best to ignore the stop if you are
        co-tuning the babbler and a top model as well.
        """
        if stop:
            int_seq = aa_seq_to_int(seq.strip())
        else:
            int_seq = aa_seq_to_int(seq.strip())[:-1]
        return int_seq


    def bucket_batch_pad(self,filepath, upper=2000, lower=50, interval=10):
        """
        Read sequences from a filepath, batch them into buckets of similar lengths, and
        pad out to the longest sequence.
        Upper, lower and interval define how the buckets are created.
        Any sequence shorter than lower will be grouped together, as with any greater 
        than upper. Interval defines the "walls" of all the other buckets.
        WARNING: Define large intervals for small datasets because the default behavior
        is to repeat the same sequence to fill a batch. If there is only one sequence
        within a bucket, it will be repeated batch_size -1 times to fill the batch.
        """
        self._bucket_upper = upper
        self._bucket_lower = lower
        self._bucket_interval = interval
        self._bucket = [self._bucket_lower + (i * self._bucket_interval) for i in range(int(self._bucket_upper / self._bucket_interval))]
        self._bucket_batch =  bucketbatchpad(
                    batch_size=self._batch_size,
                    pad_shape=([None]),
                    window_size=self._batch_size,
                    bounds=self._bucket,
                    path_to_data=filepath,
                    shuffle_buffer=self._shuffle_buffer,
                    repeat=None
        ).make_one_shot_iterator().get_next()
        return self._bucket_batch

    def split_to_tuple(self, seq_batch):
        """
        NOTICE THAT BY DEFAULT THIS STRIPS THE LAST CHARACTER.
        IF USING IN COMBINATION WITH format_seq then set stop=True there.
        Return a list of batch, target tuples.
        The input (array-like) should
        look like 
        1. . . . . . . . sequence_length
        .
        .
        .
        batch_size
        """
        q = None
        num_steps = seq_batch.shape[1]
        # Minibatches should start at zero index and go to -1
        # Don't even try to get what is happenning here its a brainfuck and
        # probably inefficient
        xypairs = [
            (seq_batch[:, :-1][:, idx:idx + num_steps], seq_batch[:, idx + 1:idx + num_steps + 1]) for idx in np.arange(len(seq_batch[0]))[0:-1:num_steps]
        ]
        if q:
            for e in xypairs:
                q.put(e)
        else:
            return xypairs[0]

    def is_valid_seq(self, seq, max_len=4000):
        """
        True if seq is valid for the babbler, False otherwise.
        """
        l = len(seq)
        ## important: added X here, not in original unirep code
        valid_aas = "MRHKDESTNQCUGPAVIFYWLOX" 
        if (l <= max_len) and set(seq) <= set(valid_aas):
            return True
        else:
            return False