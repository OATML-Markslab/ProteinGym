import os
import subprocess
import argparse
from evcouplings.couplings import CouplingsModel
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jackhmmer and EVcouplings on a query sequence")
    parser.add_argument("--query_file", default=None, help="Query sequence file")
    parser.add_argument("--database", default=None, help="Database file")
    parser.add_argument("--bitscores", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=float, nargs="+", help="Bitscore thresholds")
    parser.add_argument("--iterations", default=5, type=int, help="Number of Jackhmmer iterations")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    # 设置参数
    # query_file = "data/case/aa_seq/BFP.fasta"  # 查询序列文件
    # database = "/home/tanyang/data/0dataset/MEER/MEER_0.5.fasta"  # UniRef100数据库文件
    # output_dir = "jackhmmer_results"  # 输出目录
    # iterations = 5  # 迭代次数
    # bitscores = [round(0.1 * i, 1) for i in range(1, 10)]  # 比特得分阈值列表 [0.1, 0.2, ..., 0.9]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化变量，用于记录最佳MSA
    best_msa_file = None
    max_significant_ecs = -1

    # 遍历每个比特得分阈值
    for bit_score in args.bitscores:
        print(f"\n运行Jackhmmer，比特得分阈值：{bit_score}")
        output_prefix = f"bitscore_{bit_score}"
        alignment_sto_file = os.path.join(args.output_dir, f"{output_prefix}.sto")
        alignment_a2m_file = os.path.join(args.output_dir, f"{output_prefix}.a2m")
        tblout_file = os.path.join(args.output_dir, f"{output_prefix}_tblout.txt")
        jackhmmer_output = os.path.join(args.output_dir, f"{output_prefix}_jackhmmer.txt")
        
        # 构建Jackhmmer命令
        jackhmmer_cmd = [
            "jackhmmer",
            "-N", str(args.iterations),
            "--incT", str(bit_score),
            "--cpu", "4",
            "--tblout", tblout_file,
            "-A", alignment_sto_file,
            args.query_file,
            args.database
        ]
        
        # 运行Jackhmmer
        with open(jackhmmer_output, "w") as out_f:
            subprocess.run(jackhmmer_cmd, stdout=out_f)
        
        # 检查是否生成了对齐文件
        if not os.path.isfile(alignment_sto_file):
            print(f"比特得分 {bit_score} 下未生成对齐文件。")
            continue
        
        # 将Stockholm格式转换为A2M格式，并处理gap符号
        with open(alignment_sto_file, "r") as input_handle:
            alignments = AlignIO.read(input_handle, "stockholm")
            
            # 替换gap符号 '-' 为 '.'，符合A2M格式要求
            a2m_alignment = MultipleSeqAlignment(
                [SeqRecord(Seq(str(record.seq).replace('-', '.')), id=record.id, description="") for record in alignments]
            )
        
        with open(alignment_a2m_file, "w") as output_handle:
            AlignIO.write(a2m_alignment, output_handle, "fasta")
        
        print(f"已将对齐文件转换为A2M格式：{alignment_a2m_file}")
        
        