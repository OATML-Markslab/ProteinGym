'''
Infers log-likelihoods from UniRep models.
Source: https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data/blob/main/src/unirep_inference.py
'''

import argparse
import os
import sys
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.unirep.unirep import babbler1900
from baselines.unirep.utils import load_and_filter_seqs, save, format_batch_seqs, nonpad_len

def run_inference(seqs, model_weight_path, output_dir, output_filename=None,
        batch_size=64, save_hidden=False):
    if len(seqs) < batch_size:
        batch_size = len(seqs)
    babbler_class = babbler1900
    # Load model weights
    b = babbler_class(batch_size=batch_size, model_path=model_weight_path)
    # Load ops
    final_hidden_op, avg_hidden_op, x_ph, batch_size_ph, seq_len_ph, init_state_ph = b.get_rep_ops()
    logits_op, loss_op, x_ph, y_ph, batch_size_ph, init_state_ph = b.get_babbler_ops()
    batch_loss_op = b.batch_losses

    final_hidden_vals = []
    avg_hidden_vals = []
    loss_vals = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_batches = int(len(seqs) / batch_size)
        leftover = len(seqs) % batch_size
        n_batches += int(bool(leftover))
        for i in range(n_batches):
            print('----Running inference for batch # %d------' % i)
            if i == n_batches - 1:
                batch_seqs = seqs[-batch_size:]
            else:
                batch_seqs = seqs[i*batch_size:(i+1)*batch_size]
            batch_seqs = [seq.replace('-', 'X') for seq in batch_seqs]
            batch = format_batch_seqs(batch_seqs)
            length = nonpad_len(batch)
            # Run final hidden op
            avg_hidden_, loss_ = sess.run(
                [avg_hidden_op, batch_loss_op],
                feed_dict={
                    # Important! Shift input and expected target by 1.
                    x_ph: batch[:, :-1],
                    y_ph: batch[:, 1:],
                    batch_size_ph: batch.shape[0],
                    seq_len_ph: length,
                    init_state_ph:b._zero_state
                })
            if i == n_batches - 1:
                loss_vals.append(loss_[-leftover:])
                if save_hidden:
                    avg_hidden_vals.append(avg_hidden_[-leftover:])
            else:
                loss_vals.append(loss_)
                if save_hidden:
                    avg_hidden_vals.append(avg_hidden_)

    loss_vals = np.concatenate(loss_vals, axis=0)
    if output_filename is not None:
        output_df=pd.DataFrame({'mutated_sequence':seqs,
                                'Unirep_score':loss_vals})
        output_df.to_csv(output_dir+os.sep+output_filename+'.csv',index=False)
    #else:
    #    loss_filename = os.path.join(output_dir, f'loss.npy')
    #    save(loss_filename, loss_vals)

    #if save_hidden:
    #    avg_hidden_vals = np.concatenate(avg_hidden_vals, axis=0)
    #    avg_hidden_filename = os.path.join(output_dir, f'avg_hidden.npy')
    #    save(avg_hidden_filename, avg_hidden_vals)

    print('Ran inference on %d sequences. Saved results to %s.' %
            (len(seqs), output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--mapping_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--DMS_index', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--evotune', action='store_true')
    parser.add_argument('--save_hidden', dest='save_hidden', action='store_true')
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.mapping_path)
    list_DMS = mapping["DMS_id"]
    DMS_id=list_DMS[args.DMS_index]   
    DMS_file_name = mapping["DMS_filename"][mapping["DMS_id"]==DMS_id].values[0]
    MSA_file_name = mapping["MSA_filename"][mapping["DMS_id"]==DMS_id].values[0]
    if args.evotune:
        args.model_path = args.model_path + os.sep +  MSA_file_name.split('.a2m')[0]
    print("Computing scores for: {} with Unirep {}".format(DMS_id, args.model_path))


    seqs = load_and_filter_seqs(args.data_path + os.sep + DMS_file_name)
    #np.savetxt(os.path.join(args.output_dir, 'seqs.npy'), seqs, '%s')

    run_inference(seqs, args.model_path,
            args.output_dir, 
            output_filename=DMS_id,
            batch_size=args.batch_size,
            save_hidden=args.save_hidden)