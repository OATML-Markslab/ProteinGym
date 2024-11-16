import json
import os
import subprocess
import argparse
from tqdm import tqdm
from requests import get
from time import sleep

def process_pdb(pdb_file, out_dir):
    file_name = pdb_file.split('.')[0].split('/')[-1]
    
    if os.path.exists(f'{out_dir}/{file_name}.fasta'):
        print(f'>>> {file_name} already exists')
        return None
    
    # submit a new job and get the ticket
    repeat = True
    try_times = 0
    while repeat:
        result = subprocess.run(
            [
                "curl", "-X", "POST", "-F", f"q=@{pdb_file}", 
                "-F", "mode=3diaa", "-F", "database[]=afdb50", 
                "-F", "database[]=afdb-proteome", "-F", "database[]=cath50", 
                "-F", "database[]=mgnify_esm30", "-F", "database[]=pdb100", 
                "-F", "database[]=gmgcl_id", "-F", "database[]=afdb-swissprot", 
                "-F", "database[]=bfvd", "-F", "database[]=bfmd", 
                "https://search.foldseek.com/api/ticket"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            result = result.stdout
            ticket = json.loads(result)
            repeat = ticket['status'] != 'COMPLETE'
        except:
            sleep(1)
            try_times += 1
            print('>>> Try again for the ' + str(try_times) + ' time')
            continue
    print('>>> Ticket:', ticket)
    result = get('https://search.foldseek.com/api/result/' + ticket['id'] + '/0').json()
    
    # structure_aln_dict = json.load(open(f'structure_alignment_json/{file_name}.json'))
    structure_aln_dict = result
    # with open('result.json', 'w') as f:
    #     json.dump(result, f)
    results = structure_aln_dict['results']
    query_seq = structure_aln_dict['queries'][0]['sequence']
    
    # with open(f'residue_sequence/{file_name}.fasta', 'r') as f:
    #     seq = f.readlines()[1].strip()
    # if query_seq != seq:
    #     print(f'Error with {file_name}')
    #     return None
    
    alignment_dict = {}
    for result_db in results:
        if len(result_db['alignments']) == 0:
            continue
        for target_info in result_db['alignments'][0]:
            name = f"{target_info['target']}/prob_{target_info['prob']}/eval_{target_info['eval']}/score_{target_info['score']}/{target_info['qStartPos']}-{target_info['qEndPos']}"
            qaln = target_info['qAln']
            dbaln = target_info['dbAln']
            try:
                # get the index list of '-' in qaln
                qaln = list(qaln)
                miss_index = [i for i in range(len(qaln)) if qaln[i] == '-']
                # remove the residues in dbaln according to the missing index list in qaln
                dbaln = list(dbaln)
                dbaln = ''.join([dbaln[i] for i in range(len(dbaln)) if i not in miss_index])
            except Exception as e:
                print(e)
                print(name)
            # fill '-' to the left and right of dbaln to make it the same length as query_seq
            left = target_info['qStartPos'] - 1
            right = len(query_seq) - target_info['qEndPos']
            dbaln = '-' * left + dbaln + '-' * right
            alignment_dict[name] = dbaln

    
    if alignment_dict == {}:
        print(f'>>> {file_name} has no alignment')
        with open(f'{out_dir}/{file_name}.fasta', 'w') as f:
            f.write('\n')
        return None
    
    seqs = []
    with open(f'{out_dir}/{file_name}.fasta', 'w') as f:
        for key, value in alignment_dict.items():
            if value not in seqs:
                seqs.append(value)
            else:
                continue
            f.write(f'>{key}\n{value}\n')
    print(f'>>> {file_name} done')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str)
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    if args.pdb_dir is not None:
        pdbs = sorted(os.listdir(args.pdb_dir))
        process_bar = tqdm(pdbs)
        for pdb in process_bar:
            process_pdb(f'{args.pdb_dir}/{pdb}', args.out_dir)
            process_bar.set_description(pdb)
    elif args.pdb_file is not None:
        process_pdb(args.pdb_file, args.out_dir)