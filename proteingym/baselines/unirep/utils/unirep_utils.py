import os
import tensorflow.compat.v1 as tf

def tf_str_len(s):
    """
    Returns length of tf.string s
    """
    return tf.size(tf.string_split([s],""))

def tf_rank1_tensor_len(t):
    """
    Returns the length of a rank 1 tensor t as rank 0 int32
    """
    l = tf.reduce_sum(tf.sign(tf.abs(t)), 0)
    return tf.cast(l, tf.int32)


def tf_seq_to_tensor(s):
    """
    Input a tf.string of comma seperated integers.
    Returns Rank 1 tensor the length of the input sequence of type int32
    """
    return tf.string_to_number(
        tf.sparse_tensor_to_dense(tf.string_split([s],","), default_value='0'), out_type=tf.int32
    )[0]

def smart_length(length, bucket_bounds=tf.constant([128, 256])):
    """
    Hash the given length into the windows given by bucket bounds. 
    """
    # num_buckets = tf_len(bucket_bounds) + tf.constant(1)
    # Subtract length so that smaller bins are negative, then take sign
    # Eg: len is 129, sign = [-1,1]    
    signed = tf.sign(bucket_bounds - length)
    
    # Now make 1 everywhere that length is greater than bound, else 0
    greater = tf.sign(tf.abs(signed - tf.constant(1)))
    
    # Now simply sum to count the number of bounds smaller than length
    key = tf.cast(tf.reduce_sum(greater), tf.int64)
    
    # This will be between 0 and len(bucket_bounds)
    return key

def pad_batch(ds, batch_size, padding=None, padded_shapes=([None])):
    """
    Helper for bucket batch pad- pads with zeros
    """
    return ds.padded_batch(batch_size, 
                           padded_shapes=padded_shapes,
                           padding_values=padding
                          )

def aas_to_int_seq(aa_seq):
    int_seq = ""
    for aa in aa_seq:
        int_seq += str(aa_to_int[aa]) + ","
    return str(aa_to_int['start']) + "," + int_seq + str(aa_to_int['stop'])

# Preprocessing in python
def fasta_to_input_format(source, destination):
    # I don't know exactly how to do this in tf, so resorting to python.
    # Should go line by line so everything is not loaded into memory
    
    sourcefile = os.path.join(source)
    destination = os.path.join(destiation)
    with open(sourcefile, 'r') as f:
        with open(destination, 'w') as dest:
            seq = ""
            for line in f:
                if line[0] == '>' and not seq == "":
                    dest.write(aas_to_int_seq(seq) + '\n')
                    seq = ""
                elif not line[0] == '>':
                    seq += line.replace("\n","")

# Real data pipelines

def bucketbatchpad(
        batch_size=256,
        path_to_data=os.path.join("./data/SwissProt/sprot_ints.fasta"), # Preprocessed- see note
        compressed="", # See tf.contrib.data.TextLineDataset init args
        bounds=[128,256], # Default buckets of < 128, 128><256, >256
        # Unclear exactly what this does, should proly equal batchsize
        window_size=256, # NOT a tensor 
        padding=None, # Use default padding of zero, otherwise see Dataset docs
        shuffle_buffer=None, # None or the size of the buffer to shuffle with
        pad_shape=([None]),
        repeat=1,
        filt=None
):
    """
    Streams data from path_to_data that is correctly preprocessed.
    Divides into buckets given by bounds and pads to full length.
    Returns a dataset which will return a padded batch of batchsize
    with iteration.
    """
    batch_size=tf.constant(batch_size, tf.int64)
    bounds=tf.constant(bounds)
    window_size=tf.constant(window_size, tf.int64)
    
    path_to_data = os.path.join(path_to_data)
    # Parse strings to tensors
    dataset = tf.data.TextLineDataset(path_to_data).map(tf_seq_to_tensor)
    if filt is not None:
        dataset = dataset.filter(filt)

    if shuffle_buffer:
        # Stream elements uniformly randomly from a buffer
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Apply a repeat. Because this is after the shuffle, all elements of the dataset should be seen before repeat.
    # See https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
    dataset = dataset.repeat(count=repeat)
    # Apply grouping to bucket and pad
    group_fn = tf.data.experimental.group_by_window(
        key_func=lambda seq: smart_length(tf_rank1_tensor_len(seq), bucket_bounds=bounds), # choose a bucket
        reduce_func=lambda key, ds: pad_batch(ds, batch_size, padding=padding, padded_shapes=pad_shape), # apply reduce funtion to pad
        window_size=window_size)
    grouped_dataset = dataset.apply(group_fn)
    return grouped_dataset

def shufflebatch(
        batch_size=256,
        shuffle_buffer=None,
        repeat=1,
        path_to_data="./data/SwissProt/sprot_ints.fasta"
):
    """
    Draws from an (optionally shuffled) dataset, repeats dataset repeat times,
    and serves batches of the specified size.
    """
    
    path_to_data = os.path.join(path_to_data)
    # Parse strings to tensors
    dataset = tf.contrib.data.TextLineDataset(path_to_data).map(tf_seq_to_tensor)
    if shuffle_buffer:
        # Stream elements uniformly randomly from a buffer
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Apply a repeat. Because this is after the shuffle, all elements of the dataset should be seen before repeat.
    # See https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
    dataset = dataset.repeat(count=repeat)
    dataset = dataset.batch(batch_size)
    return dataset