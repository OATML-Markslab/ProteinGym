protein_name=<your_protein_name>
protein_path=<your_protein_path>
database=<your_path>/uniref100.fasta
evcouplings \
    -P output/$protein/$protein \
    -p $protein_name \
    -s $protein_path \
    -d $database \
    -b "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
    -n 5 src/single_config_monomer.txt
