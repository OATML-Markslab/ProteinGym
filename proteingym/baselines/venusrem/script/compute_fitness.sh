# zero-shot with aa sequence alignment
export HF_ENDPOINT=https://hf-mirror.com
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/$protein_dir


# zero-shot with structure sequence alignment
export HF_ENDPOINT=https://hf-mirror.com
alpha=0.8
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --logit_mode struc_seq_aln \
    --alpha $alpha \
    --model_out_name ProtREM-struc \
    --base_dir data/$protein_dir \
    --out_scores_dir result/$protein_dir