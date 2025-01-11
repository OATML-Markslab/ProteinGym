import os
import argparse
import shutil
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='data/MSA')
    parser.add_argument('--is_multi', action='store_true')
    parser.add_argument('-o', '--output_dir', type=str, default='data/MSA_selected')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir + '/aa_seq_aln_a2m_raw', exist_ok=True)
    os.makedirs(args.output_dir + '/aa_seq_aln_a2m', exist_ok=True)
    
    if args.is_multi:
        failed_jobs = []
        protein_names = os.listdir(args.input_dir)
        for protein_name in tqdm(protein_names):
            job_summary = pd.read_csv(f'{args.input_dir}/{protein_name}/{protein_name}_job_statistics_summary.csv')
            if len(job_summary) != 9:
                print(f'>>> No enough MSA for {protein_name}')
                files = os.listdir(f'{args.input_dir}/{protein_name}')
                failed_files = [f for f in files if f.endswith('failed')]
                if len(failed_files) + len(job_summary) != 9:
                    print(f'>>> Not all jobs are finished for {protein_name}, skippping...')
                    failed_jobs.append(protein_name)
                    continue
            
            # select the `prefix` according to the max `num_significant`
            prefix = job_summary.loc[job_summary['num_significant'].idxmax()]['prefix']
            prefix_name = prefix.split('/')[-1]
            shutil.copyfile(f'{args.input_dir}/{protein_name}/{prefix_name}/align/{prefix_name}_raw_focus.fasta', f'{args.output_dir}/aa_seq_aln_a2m_raw/{protein_name}.fasta')
            shutil.copyfile(f'{args.input_dir}/{protein_name}/{prefix_name}/align/{prefix_name}.a2m', f'{args.output_dir}/aa_seq_aln_a2m/{protein_name}.a2m')
            print(f'>>> {protein_name} selected, prefix: {prefix_name}')
    else:
        protein_name = args.input_dir.split('/')[-1]
        job_summary = pd.read_csv(f'{args.input_dir}/{protein_name}_job_statistics_summary.csv')
        if len(job_summary) != 9:
            print(f'>>> No enough MSA for {protein_name}')
        # select the `prefix` according to the max `num_significant`
        prefix = job_summary.loc[job_summary['num_significant'].idxmax()]['prefix']
        prefix_name = prefix.split('/')[-1]
        shutil.copyfile(f'{args.input_dir}/{prefix_name}/align/{prefix_name}_raw_focus.fasta', f'{args.output_dir}/aa_seq_aln_a2m_raw/{protein_name}.fasta')
        shutil.copyfile(f'{args.input_dir}/{prefix_name}/align/{prefix_name}.a2m', f'{args.output_dir}/aa_seq_aln_a2m/{protein_name}.a2m')
        print(f'>>> {protein_name} selected, prefix: {prefix_name}')