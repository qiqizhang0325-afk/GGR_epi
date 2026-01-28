"""
完整的全基因组 embedding 脚本（reference + per-sample mutant）
- 基因区 + 基因间区
- PlantCAD2: single-nucleotide tokenization
- 8192bp window + 4096bp stride
- Attention pooling 融合窗口
- 变异注入：仅 SNP（基于 VCF，600+ 样本）
"""

import gc
import os
import time

import pickle
import warnings
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import argparse

from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from cyvcf2 import VCF

from bisect import bisect_left
from collections import defaultdict


# 假设 DataValidator 类已经定义


warnings.filterwarnings('ignore')


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =====================
#   GFF 基因数据库
# =====================


# vcf_file='/lustre/BIF/nobackup/zhang479/genomes/arabidopsis/test.vcf'
# gff_file='/lustre/BIF/nobackup/zhang479/genomes/arabidopsis/TAIR10_GFF3_genes.gff'
# fasta_file='/lustre/BIF/nobackup/zhang479/genomes/arabidopsis/TAIR10_chr_all.fas'


class GeneDatabase:
    """基因信息数据库 - 从 GFF 文件构建"""

    def __init__(self, gff_file: str):
        self.genes = self._parse_gff(gff_file)

    def _parse_gff(self, gff_file: str) -> dict:
        print(f"📁 解析 GFF 文件: {gff_file}")
        genes = {}

        try:
            with open(gff_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue

                    chrom = parts[0]
                    feature = parts[2]
                    start = int(parts[3])      # 1-based
                    end = int(parts[4])        # inclusive
                    strand = parts[6]
                    attributes = parts[8]

                    # 简单规范：纯数字前面补 Chr
                    if not chrom.startswith('Chr') and chrom.isdigit():
                        chrom = f'Chr{chrom}'

                    if feature != 'gene':
                        continue

                    gene_id = self._extract_gene_id(attributes)
                    if gene_id:
                        genes[gene_id] = {
                            'chrom': chrom,
                            'start': start,
                            'end': end,
                            'strand': strand
                        }
        except Exception as e:
            print(f"❌ GFF 文件解析错误: {e}")
            raise

        print(f"✅ 找到 {len(genes)} 个基因")
        return genes

    def _extract_gene_id(self, attributes: str) -> Optional[str]:
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('ID='):
                return attr.split('ID=')[1].split(',')[0]
            elif attr.startswith('gene='):
                return attr.split('gene=')[1].split(',')[0]
        return None

    def get_gene_info(self, gene_id: str) -> Optional[dict]:
        return self.genes.get(gene_id)


# =====================
#   输入验证 & 染色体标准化
# =====================

'''
class DataValidator:
    @staticmethod
    def ensure_bgzip_and_index(vcf_file: str) -> str:
        vcf = Path(vcf_file)
        if vcf.suffix == ".gz":
            tbi = str(vcf) + ".tbi"
            if not os.path.exists(tbi):
                print(f"🔧 缺失索引，正在创建: {tbi}")
                subprocess.run(["tabix", "-p", "vcf", str(vcf)], check=True)
            return str(vcf)

        gz = str(vcf) + ".gz"
        tbi = gz + ".tbi"
        if not os.path.exists(gz):
            print(f"🔧 bgzip 压缩 VCF → {gz}")
            with open(gz, "wb") as out_f:
                subprocess.run(["bgzip", "-c", str(vcf)], stdout=out_f, check=True)
        if not os.path.exists(tbi):
            print(f"🔧 创建索引: {tbi}")
            subprocess.run(["tabix", "-p", "vcf", gz], check=True)
        return gz

    @staticmethod
    def validate_inputs(fasta_file: str, vcf_file: str, gff_file: str) -> bool:
        print("\n🔍 验证输入文件...")
        all_valid = True
        for file_path, file_type in [(fasta_file, "FASTA"),
                                     (vcf_file, "VCF"),
                                     (gff_file, "GFF")]:
            if not Path(file_path).exists():
                print(f"❌ {file_type} 文件不存在: {file_path}")
                all_valid = False
            else:
                size_gb = Path(file_path).stat().st_size / (1024**3)
                print(f"✅ {file_type}: {size_gb:.2f} GB")
        return all_valid

    @staticmethod
    def standardize_chrom_name(chrom: str) -> str:
        return chrom.replace("Chr", "").replace("chr", "").upper()

    @staticmethod
    def normalize_chromosome_names(genome_keys: List[str],
                                   vcf_chroms: List[str],
                                   gff_chroms: List[str]) -> Dict[str, Dict[str, str]]:
        print("\n🔧 标准化染色体名称...")

        genome_map = {DataValidator.standardize_chrom_name(c): c for c in genome_keys}
        vcf_map = {DataValidator.standardize_chrom_name(c): c for c in vcf_chroms}
        gff_map = {DataValidator.standardize_chrom_name(c): c for c in gff_chroms}

        all_keys = set(genome_map) | set(vcf_map) | set(gff_map)

        normalized_mapping = {}
        for k in sorted(all_keys):
            normalized_mapping[k] = {
                'fasta': genome_map.get(k),
                'vcf': vcf_map.get(k),
                'gff': gff_map.get(k)
            }

        print(f"✅ 标准化完成，检测到 {len(normalized_mapping)} 条染色体映射")
        for ek in list(normalized_mapping.keys())[:5]:
            print(f"   {ek}: {normalized_mapping[ek]}")

        return normalized_mapping
'''



class DataValidator:
    @staticmethod
    def ensure_bgzip_and_index(vcf_file: str) -> str:
        """
        确保 VCF 文件是 BGZIP 压缩且拥有 tabix 索引。
        如果文件是普通 GZIP 压缩，则先解压再用 bgzip 重新压缩。
        返回最终的 BGZIP 文件路径。
        """
        vcf_path = Path(vcf_file)
        
        # 1. 如果文件是 .gz 格式
        if vcf_path.suffix == ".gz":
            # 假设文件已经是 BGZIP，先尝试直接索引
            print(f"🔄 检查压缩 VCF 文件: {vcf_path.name}")
            gz_file = str(vcf_path)
            tbi_file = gz_file + ".tbi"

            if os.path.exists(tbi_file):
                print(f"✅ 索引已存在: {tbi_file}")
                return gz_file

            print(f"🔧 尝试创建索引: {tbi_file}")
            
            try:
                # 尝试对现有 .gz 文件创建索引
                subprocess.run(
                    ["tabix", "-p", "vcf", gz_file], 
                    check=True,  # 如果 tabix 失败，将抛出 CalledProcessError
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                print("✅ 索引创建成功，文件为 BGZIP 格式。")
                return gz_file
            
            except subprocess.CalledProcessError as e:
                # 如果 tabix 失败，通常意味着它不是 BGZF 文件
                print(f"⚠️ tabix 失败，文件可能不是 BGZIP 格式。错误信息（部分）：{e.stderr.decode().strip()}")
                
                # 创建临时文件用于解压操作
                uncompressed_vcf = vcf_path.with_suffix('')
                
                # --- A. 解压文件 ---
                print(f"🔧 假定为普通 GZIP，正在解压 → {uncompressed_vcf.name}")
                try:
                    # 使用 gzip -d (或直接调用 gunzip) 进行解压
                    subprocess.run(["gzip", "-d", "-f", gz_file], check=True)
                except subprocess.CalledProcessError:
                    print("❌ 解压失败，请检查文件权限或格式。")
                    raise

                # --- B. 重新执行 BGZIP 压缩和索引流程 ---
                # 此时 uncompressed_vcf 应该存在
                vcf_path = uncompressed_vcf


        # 2. 文件是未压缩的 VCF 文件 (或刚从 GZIP 中解压出来)
        
        # 目标 BGZIP 文件名
        gz_file = str(vcf_path) + ".gz"
        tbi_file = gz_file + ".tbi"

        # --- C. 压缩文件 ---
        needs_compression = not os.path.exists(gz_file) or (os.path.exists(gz_file) and os.path.exists(vcf_path))

        if needs_compression:
            if os.path.exists(gz_file):
                print(f"⚠️ 发现已存在的 {Path(gz_file).name}，但原始VCF文件仍存在，重新执行 BGZIP 压缩。")
                os.remove(gz_file) # 删除旧的（可能是普通 GZIP 或不完整的）压缩文件
                # 如果索引存在，也删除，确保重新索引
                if os.path.exists(tbi_file):
                    os.remove(tbi_file)
            
            print(f"🔧 BGZIP 压缩 VCF → {Path(gz_file).name}")
            try:
                # 使用 bgzip -c 压缩并重定向输出到新文件
                with open(gz_file, "wb") as out_f:
                    subprocess.run(["bgzip", "-c", str(vcf_path)], stdout=out_f, check=True)
                # 压缩成功后，删除原始未压缩文件
                os.remove(vcf_path) 
            except subprocess.CalledProcessError:
                print("❌ BGZIP 压缩失败，请检查 bgzip 是否可用。")
                raise
        else:
            print(f"✅ BGZIP 文件已存在: {Path(gz_file).name}")


        # --- D. 创建索引 (现在我们确定 gz_file 应该是一个 BGZIP 文件) ---
        if not os.path.exists(tbi_file):
            print(f"🔧 创建索引: {Path(tbi_file).name}")
            try:
                # 对新创建的 BGZIP 文件创建索引
                subprocess.run(["tabix", "-p", "vcf", gz_file], check=True)
                print("✅ 索引创建成功。")
            except subprocess.CalledProcessError as e:
                # 再次失败，这才是真正的问题，可能是 tabix/vcf 文件内容问题
                print("❌ Tabix 索引创建失败。")
                # 针对您的情况：如果 Tabix 失败，这可能仍然是文件格式问题，
                # 但既然我们刚刚用 bgzip 压缩了，更可能是 VCF 格式本身的问题（比如未排序）。
                print(f"致命错误：虽然刚刚执行了 BGZIP 压缩，但 Tabix 仍失败。请检查 VCF 文件是否已排序。")
                # 此时，应该让用户看到原始的 tabix 错误信息
                if e.stderr:
                    print(f"Tabix 错误输出: {e.stderr.decode().strip()}")
                raise
        else:
            print(f"✅ 索引文件已存在: {Path(tbi_file).name}")
            
        return gz_file

    @staticmethod
    def validate_inputs(fasta_file: str, vcf_file: str, gff_file: str) -> bool:
        print("\n🔍 验证输入文件...")
        all_valid = True
        for file_path, file_type in [(fasta_file, "FASTA"),
                                     (vcf_file, "VCF"),
                                     (gff_file, "GFF")]:
            if not Path(file_path).exists():
                print(f"❌ {file_type} 文件不存在: {file_path}")
                all_valid = False
            else:
                size_gb = Path(file_path).stat().st_size / (1024**3)
                print(f"✅ {file_type}: {size_gb:.2f} GB")
        return all_valid

    @staticmethod
    def standardize_chrom_name(chrom: str) -> str:
        return chrom.replace("Chr", "").replace("chr", "").upper()

    @staticmethod
    def normalize_chromosome_names(genome_keys: List[str],
                                   vcf_chroms: List[str],
                                   gff_chroms: List[str]) -> Dict[str, Dict[str, str]]:
        print("\n🔧 标准化染色体名称...")

        genome_map = {DataValidator.standardize_chrom_name(c): c for c in genome_keys}
        vcf_map = {DataValidator.standardize_chrom_name(c): c for c in vcf_chroms}
        gff_map = {DataValidator.standardize_chrom_name(c): c for c in gff_chroms}

        all_keys = set(genome_map) | set(vcf_map) | set(gff_map)

        normalized_mapping = {}
        for k in sorted(all_keys):
            normalized_mapping[k] = {
                'fasta': genome_map.get(k),
                'vcf': vcf_map.get(k),
                'gff': gff_map.get(k)
            }

        print(f"✅ 标准化完成，检测到 {len(normalized_mapping)} 条染色体映射")
        for ek in list(normalized_mapping.keys())[:5]:
            print(f"   {ek}: {normalized_mapping[ek]}")

        return normalized_mapping



# =====================
#   VCF 包装
# =====================

class PopulationVCF:
    """
    群体 VCF 封装
    - 只处理 SNP
    - 提供样本列表 + cyvcf2.VCF 对象 + sample_index
    """

    def __init__(self, vcf_file: str):
        print(f"\n📥 加载 VCF: {vcf_file}")
        self.vcf = VCF(vcf_file)
        self.samples = self.vcf.samples
        self.sample_index = {s: i for i, s in enumerate(self.samples)}
        print(f"📊 VCF 包含 {len(self.samples)} 个样本")

    def get_chroms(self) -> List[str]:
        return list(self.vcf.seqnames)


# =====================
#   序列质量检查
# =====================

class SequenceQualityChecker:
    @staticmethod
    def validate_dna_sequence(sequence: str) -> tuple:
        sequence = sequence.upper()

        valid_bases = set('ATCGN')
        invalid_bases = set(sequence) - valid_bases
        if invalid_bases:
            print(f"⚠️  发现无效碱基 {invalid_bases}")
            for base in invalid_bases:
                sequence = sequence.replace(base, 'N')

        if len(sequence) < 50:
            print(f"⚠️  序列过短 ({len(sequence)} bp)")

        if len(sequence) > 8192:
            print(f"ℹ️  序列过长 ({len(sequence)} bp)，将被截断到8192bp")
            sequence = sequence[:8192]

        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100

        return sequence, gc_content

    @staticmethod
    def get_sequence_stats(sequence: str) -> Dict:
        sequence = sequence.upper()
        total_len = len(sequence)
        return {
            'length': total_len,
            'gc_content': (sequence.count('G') + sequence.count('C')) / total_len * 100 if total_len > 0 else 0,
            'n_content': sequence.count('N') / total_len * 100 if total_len > 0 else 0,
            'valid_ratio': len(set(sequence) & set('ATCGN')) / total_len * 100 if total_len > 0 else 0
        }


# =====================
#   PlantCAD2 单序列 embedder
# =====================

class PlantCAD2GeneEmbedder:
    """
    官方风格的 PlantCAD2 单序列 embedding
    - single-nucleotide token
    - MLM hidden_states[-1]
    - forward + RC 融合
    """

    def __init__(self, model_name='kuleshov-group/PlantCAD2-Large-l48-d1536',
                 device='cuda:0'):
        print(f"\n🤖 加载 PlantCAD2 模型: {model_name}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        self.config = self.model.config
        self.embedding_dim = getattr(self.config, "hidden_size", 1536)
        self.bidirectional_strategy = getattr(self.config, "bidirectional_strategy", "add")

        print(f"📏 Embedding dim = {self.embedding_dim}")
        print(f"🔄 Bidirectional strategy = {self.bidirectional_strategy}")

    def reverse_complement(self, seq: str) -> str:
        table = str.maketrans("ATCGN", "TAGCN")
        return seq.upper().translate(table)[::-1]

    def _pool(self, x: torch.Tensor, strategy="mean"):
        if strategy == "mean":
            return x.mean(dim=1).squeeze(0)
        if strategy == "max":
            return x.max(dim=1).values.squeeze(0)
        if strategy == "first":
            return x[:, 0, :].squeeze(0)
        if strategy == "last":
            return x[:, -1, :].squeeze(0)
        return x.mean(dim=1).squeeze(0)

    def get_single_embedding(self, sequence: str, pooling_strategy: str = 'mean') -> np.ndarray:
        sequence = sequence.upper()
        if len(sequence) > 8192:
            sequence = sequence[:8192]
        elif len(sequence) < 50:
            print(f"⚠️  序列过短 ({len(sequence)} bp)")

        # forward
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            out_fwd = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            emb_fwd = out_fwd.hidden_states[-1]

        # reverse-complement
        rc_seq = self.reverse_complement(sequence)
        rc_inputs = self.tokenizer(
            rc_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        rc_ids = rc_inputs["input_ids"].to(self.device)

        with torch.no_grad():
            out_rc = self.model(
                input_ids=rc_ids,
                output_hidden_states=True,
                return_dict=True
            )
            emb_rc = out_rc.hidden_states[-1]
            emb_rc = torch.flip(emb_rc, [1])

        if self.bidirectional_strategy == "add":
            emb = (emb_fwd + emb_rc) / 2
        elif self.bidirectional_strategy == "ew_multiply":
            emb = emb_fwd * emb_rc
        else:
            emb = (emb_fwd + emb_rc) / 2

        pooled = self._pool(emb, pooling_strategy)
        return pooled.cpu().numpy()


# =====================
#   Attention Pool + Locus 级长窗口 embedding
# =====================

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [num_windows, dim]
        """
        scores = torch.softmax(self.att(x), dim=0)  # [num_windows, 1]
        return torch.sum(scores * x, dim=0)         # [dim]


class LocusEmbedderLongWindow:
    """
    统一的 locus embedding（基因区 & 基因间区）：
    - 任意长度序列
    - 8192bp window + 4096bp stride
    - Attention pooling 融合多窗口
    """

    def __init__(self, plantcad_embedder: PlantCAD2GeneEmbedder,
                 window_size: int = 8192,
                 stride: int = 4096,
                 pooling: str = "mean"):
        self.embedder = plantcad_embedder
        self.window_size = window_size
        self.stride = stride
        self.pooling = pooling
        self.att_pool = AttentionPool(self.embedder.embedding_dim)

    def _split_windows(self, seq: str) -> List[str]:
        seq = seq.upper()
        L = len(seq)
        ws = self.window_size
        st = self.stride

        if L <= ws:
            return [seq]

        windows = []
        for i in range(0, L - ws + 1, st):
            windows.append(seq[i:i + ws])

        if (L - ws) % st != 0:
            windows.append(seq[-ws:])

        return windows

    def embed_locus(self, seq: str) -> np.ndarray:
        windows = self._split_windows(seq)
        window_embs = []

        for w in windows:
            emb = self.embedder.get_single_embedding(w, pooling_strategy=self.pooling)
            window_embs.append(torch.tensor(emb))

        if len(window_embs) == 1:
            return window_embs[0].numpy()

        window_embs = torch.stack(window_embs)
        locus_emb = self.att_pool(window_embs).detach().numpy()
        return locus_emb


class LocusEmbedderAdaptive:
    """
    基于序列长度的自适应 locus embedder：
    - <= short_threshold: 单窗口（无 attention pooling）
    - > short_threshold: long-window + attention pooling
    """

    def __init__(
        self,
        plantcad_embedder: PlantCAD2GeneEmbedder,
        short_threshold: int = 4096,
        window_size: int = 4096,
        stride: int = 2048,
        pooling: str = "mean"
    ):
        self.embedder = plantcad_embedder
        self.short_threshold = short_threshold
        self.window_size = window_size
        self.stride = stride
        self.pooling = pooling
        self.att_pool = AttentionPool(self.embedder.embedding_dim)

    def _split_windows(self, seq: str):
        L = len(seq)
        ws = self.window_size
        st = self.stride

        windows = []
        for i in range(0, L - ws + 1, st):
            windows.append(seq[i:i + ws])
        if (L - ws) % st != 0:
            windows.append(seq[-ws:])
        return windows

    def embed_locus(self, seq: str) -> np.ndarray:
        L = len(seq)

        # ===== 1️⃣ 短 gene：直接 single forward =====
        if L <= self.short_threshold:
            return self.embedder.get_single_embedding(
                seq,
                pooling_strategy=self.pooling
            )

        # ===== 2️⃣ 长 gene：long-window + attention =====
        windows = self._split_windows(seq)
        window_embs = []

        for w in windows:
            emb = self.embedder.get_single_embedding(
                w,
                pooling_strategy=self.pooling
            )
            window_embs.append(torch.tensor(emb))

        window_embs = torch.stack(window_embs)  # [num_windows, dim]
        locus_emb = self.att_pool(window_embs)
        return locus_emb.detach().cpu().numpy()



class SNPIndexBuilder:
    """
    为单个 sample 构建 SNP 位置索引：
    chrom -> sorted positions
    """

    def __init__(self, pop_vcf, seq_builder):
        self.pop_vcf = pop_vcf
        self.seq_builder = seq_builder

    def build_for_sample(self, sample_name: str):
        snp_index = defaultdict(list)

        if self.pop_vcf is None:
            return snp_index

        if sample_name not in self.pop_vcf.sample_index:
            return snp_index

        s_idx = self.pop_vcf.sample_index[sample_name]

        for rec in self.pop_vcf.vcf:
            # 只保留 SNP
            if len(rec.REF) != 1 or not rec.ALT or len(rec.ALT[0]) != 1:
                continue

            gts = rec.genotypes
            if s_idx >= len(gts):
                continue

            gt = gts[s_idx][:2]
            if gt[0] is None or gt[1] is None:
                continue

            if 1 not in gt:
                continue

            fasta_chrom = self.seq_builder._get_actual_chromosome(rec.CHROM)
            snp_index[fasta_chrom].append(rec.POS)

        # 排序（为二分查找）
        for chrom in snp_index:
            snp_index[chrom].sort()

        return snp_index

    def build_for_sample_chrom(self, sample_name: str, fasta_chrom: str, chrom_len: int):
        snp_index = defaultdict(list)
        if self.pop_vcf is None:
            return snp_index

        vcf_chrom = self.seq_builder._get_vcf_chrom_from_fasta(fasta_chrom)
        region = f"{vcf_chrom}:1-{chrom_len}"

        s_idx = self.pop_vcf.sample_index.get(sample_name, None)
        if s_idx is None:
            return snp_index

        for rec in self.pop_vcf.vcf(region):   # ✅ 只迭代这一条染色体（tabix）
            if len(rec.REF) != 1 or not rec.ALT or len(rec.ALT[0]) != 1:
                continue
            gt = rec.genotypes[s_idx][:2]
            if gt[0] is None or gt[1] is None:
                continue
            if 1 in gt:
                snp_index[fasta_chrom].append(rec.POS)

        snp_index[fasta_chrom].sort()
        return snp_index




# =====================
#   序列构建（reference + mutant）
# =====================

class SequenceBuilder:
    """
    构建参考 & 样本特异序列：
    - 基因区：通过 gene_db + GFF
    - 基因间区：通过 intergenic region 坐标
    - mutant：使用 PopulationVCF 在对应 region 注入 SNP
    """

    def __init__(self, genome: Dict, gene_db: GeneDatabase,
                 pop_vcf: Optional[PopulationVCF] = None,
                 chrom_mapping: Optional[Dict[str, Dict[str, str]]] = None):
        self.genome = genome
        self.gene_db = gene_db
        self.pop_vcf = pop_vcf
        self.chrom_mapping = chrom_mapping

    def _get_actual_chromosome(self, gff_chrom: str) -> str:
        """
        GFF 染色体名 → FASTA 染色体名
        """
        if self.chrom_mapping:
            for std, m in self.chrom_mapping.items():
                if m.get('gff') == gff_chrom or m.get('fasta') == gff_chrom:
                    fasta_chrom = m.get('fasta')
                    if fasta_chrom:
                        return fasta_chrom
        return gff_chrom

    def _get_vcf_chrom_from_fasta(self, fasta_chrom: str) -> str:
        """
        FASTA 染色体名 → VCF 染色体名
        """
        if self.chrom_mapping:
            for std, m in self.chrom_mapping.items():
                if m.get('fasta') == fasta_chrom:
                    vcf_chrom = m.get('vcf')
                    if vcf_chrom:
                        return vcf_chrom
        return fasta_chrom

    # ---------- 参考序列 ----------

    def build_reference_sequence(self, gene_id: str, flank: int = 0) -> Optional[str]:
        info = self.gene_db.get_gene_info(gene_id)
        if not info:
            return None

        fasta_chrom = self._get_actual_chromosome(info['chrom'])
        if fasta_chrom not in self.genome:
            print(f"⚠️  染色体 {fasta_chrom} 在 FASTA 中未找到")
            return None

        start = max(1, info['start'] - flank)
        end = min(len(self.genome[fasta_chrom]), info['end'] + flank)

        seq = str(self.genome[fasta_chrom].seq[start - 1:end]).upper()
        if info['strand'] == '-':
            seq = str(Seq(seq).reverse_complement())
        return seq

    # ---------- mutant 通用 locus 序列 ----------

    def build_sample_locus_sequence(
        self,
        fasta_chrom: str,
        start: int,
        end: int,
        sample_name: str,
        strand: Optional[str] = None
    ) -> Optional[str]:
        """
        构建某个 locus（基因 or 间区）在某个样本下的 mutant 序列（只注入 SNP）
        start/end: 1-based inclusive
        """
        if fasta_chrom not in self.genome:
            return None
        if self.pop_vcf is None:
            # 没有 VCF，就退化为 reference
            seq = str(self.genome[fasta_chrom].seq[start - 1:end]).upper()
            if strand == '-':
                seq = str(Seq(seq).reverse_complement())
            return seq

        ref_seq = str(self.genome[fasta_chrom].seq[start - 1:end]).upper()
        if len(ref_seq) == 0:
            return None

        seq_list = list(ref_seq)

        vcf_chrom = self._get_vcf_chrom_from_fasta(fasta_chrom)

        if sample_name not in self.pop_vcf.sample_index:
            return None
        s_idx = self.pop_vcf.sample_index[sample_name]

        region = f"{vcf_chrom}:{start}-{end}"
        try:
            for rec in self.pop_vcf.vcf(region):
                pos = rec.POS
                ref = rec.REF
                alt = rec.ALT[0] if rec.ALT else None

                # 只处理 SNP
                if len(ref) != 1 or (not alt or len(alt) != 1):
                    continue

                gts = rec.genotypes
                if s_idx >= len(gts):
                    continue
                gt = gts[s_idx][:2]
                if gt[0] is None or gt[1] is None:
                    continue

                # 携带 ALT 等位基因
                if 1 in gt:
                    rel_pos = pos - start  # 0-based 相对坐标
                    if 0 <= rel_pos < len(seq_list):
                        seq_list[rel_pos] = alt
        except Exception as e:
            print(f"⚠️  在 region {region} 读取 VCF 时出错: {e}")

        sample_seq = ''.join(seq_list)
        if strand == '-':
            sample_seq = str(Seq(sample_seq).reverse_complement())
        return sample_seq

    # ---------- mutant 基因序列 ----------

    def build_sample_gene_sequence(self, gene_id: str, sample_name: str, flank: int = 0) -> Optional[str]:
        info = self.gene_db.get_gene_info(gene_id)
        if not info:
            return None

        fasta_chrom = self._get_actual_chromosome(info['chrom'])
        if fasta_chrom not in self.genome:
            return None

        start = max(1, info['start'] - flank)
        end = min(len(self.genome[fasta_chrom]), info['end'] + flank)
        strand = info['strand']

        return self.build_sample_locus_sequence(
            fasta_chrom=fasta_chrom,
            start=start,
            end=end,
            sample_name=sample_name,
            strand=strand
        )

    def build_all_samples_for_gene(self, gene_id: str, flank: int = 0,
                                   sample_subset: Optional[List[str]] = None) -> Dict[str, str]:
        sequences = {}
        if self.pop_vcf is None:
            return sequences

        if sample_subset:
            target_samples = [s for s in sample_subset if s in self.pop_vcf.samples]
        else:
            target_samples = self.pop_vcf.samples

        for s in target_samples:
            seq = self.build_sample_gene_sequence(gene_id, s, flank)
            if seq:
                sequences[s] = seq
        return sequences

    def region_has_snp(
        self,
        fasta_chrom: str,
        start: int,
        end: int,
        sample_name: str
    ) -> bool:
        """
        判断某个样本在指定 region 是否至少有一个 SNP
        """
        if self.pop_vcf is None:
            return False

        if sample_name not in self.pop_vcf.sample_index:
            return False

        vcf_chrom = self._get_vcf_chrom_from_fasta(fasta_chrom)
        s_idx = self.pop_vcf.sample_index[sample_name]

        region = f"{vcf_chrom}:{start}-{end}"

        try:
            for rec in self.pop_vcf.vcf(region):
                # 只关心 SNP
                if len(rec.REF) != 1 or not rec.ALT or len(rec.ALT[0]) != 1:
                    continue

                gts = rec.genotypes
                if s_idx >= len(gts):
                    continue

                gt = gts[s_idx][:2]
                if gt[0] is None or gt[1] is None:
                    continue

                # 样本携带 ALT
                if 1 in gt:
                    return True
        except Exception:
            return False

        return False



# =====================
#   基因间区构建
# =====================

def build_intergenic_regions(gene_db: GeneDatabase,
                             genome: Dict,
                             chrom_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> List[dict]:
    print("\n🧱 构建基因间区 (intergenic regions)...")

    gff_to_std = {}
    fasta_len = {}

    if chrom_mapping:
        for std, m in chrom_mapping.items():
            gff_ch = m.get('gff')
            fa_ch = m.get('fasta')
            if gff_ch:
                gff_to_std[gff_ch] = std
            if fa_ch and fa_ch in genome:
                fasta_len[std] = len(genome[fa_ch])

    chrom_to_genes = {}
    for gid, info in gene_db.genes.items():
        gff_chrom = info['chrom']
        std_chrom = gff_to_std.get(gff_chrom, gff_chrom)
        chrom_to_genes.setdefault(std_chrom, []).append((gid, info))

    intergenic_regions = []

    for std_chrom, g_list in chrom_to_genes.items():
        fasta_chrom = None
        if chrom_mapping and std_chrom in chrom_mapping:
            fasta_chrom = chrom_mapping[std_chrom].get('fasta')
        else:
            fasta_chrom = std_chrom

        if fasta_chrom not in genome:
            print(f"⚠️  标准染色体 {std_chrom} 对应的 FASTA 染色体 {fasta_chrom} 未找到，跳过")
            continue

        chrom_len = len(genome[fasta_chrom])
        g_list.sort(key=lambda x: x[1]['start'])

        first_start = g_list[0][1]['start']
        if first_start > 1:
            intergenic_regions.append({
                'chrom_std': std_chrom,
                'chrom_fasta': fasta_chrom,
                'start': 1,
                'end': first_start - 1,
                'id': f'{std_chrom}_intergenic_0000'
            })

        for i in range(len(g_list) - 1):
            cur_end = g_list[i][1]['end']
            next_start = g_list[i + 1][1]['start']
            if next_start > cur_end + 1:
                intergenic_regions.append({
                    'chrom_std': std_chrom,
                    'chrom_fasta': fasta_chrom,
                    'start': cur_end + 1,
                    'end': next_start - 1,
                    'id': f'{std_chrom}_intergenic_{i + 1:04d}'
                })

        last_end = g_list[-1][1]['end']
        if last_end < chrom_len:
            intergenic_regions.append({
                'chrom_std': std_chrom,
                'chrom_fasta': fasta_chrom,
                'start': last_end + 1,
                'end': chrom_len,
                'id': f'{std_chrom}_intergenic_tail'
            })

    print(f"✅ 共构建 {len(intergenic_regions)} 个 intergenic loci")
    return intergenic_regions


# =====================
#   工具函数
# =====================

def get_directory_size(path: str) -> float:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)


def region_has_snp_fast(
    snp_index: dict,
    chrom: str,
    start: int,
    end: int
) -> bool:
    """
    O(log N) 判断 region 是否包含 SNP
    """
    if chrom not in snp_index:
        return False

    positions = snp_index[chrom]
    i = bisect_left(positions, start)
    return i < len(positions) and positions[i] <= end

# =====================
#   主流程
# =====================

def main():
    parser = argparse.ArgumentParser(description='Genome-wide reference + per-sample mutant embedding with PlantCAD2')

    parser.add_argument('--fasta', required=True, help='参考基因组 FASTA 文件')
    parser.add_argument('--vcf', required=True, help='群体 VCF 文件')
    parser.add_argument('--gff', required=True, help='GFF 基因注释文件')
    parser.add_argument('--output', default='./genome_embeddings', help='输出目录')
    parser.add_argument('--model', default='kuleshov-group/PlantCAD2-Large-l48-d1536')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--flank', type=int, default=0, help='基因两侧延伸碱基数')
    parser.add_argument('--gene_list', type=str,
                        help='包含 gene ID 的文件（每行一个）；若提供，仅处理其中的基因')
    parser.add_argument('--max_genes', type=int, help='最大处理基因数（用于测试）')
    parser.add_argument('--samples', nargs='*',
                        help='可选：只处理指定样本名；默认 VCF 中全部样本')

    parser.add_argument(
        '--mode',
        choices=['reference', 'sample', 'all'],
        default='all',
        help='run mode: '
            'reference=only reference, '
            'sample=only sample (requires existing reference), '
            'all=reference+sample (not recommended for multi-GPU)'
    )

    parser.add_argument(
        '--chrom',
        type=str,
        default=None,
        help='只处理某条 FASTA 染色体（例如 Chr1）；不指定则全基因组'
    )

    args = parser.parse_args()
    selected_chrom = args.chrom  # e.g. "Chr1" in FASTA naming


    print("=" * 80)
    print("🧬 全基因组 PlantCAD2 Embedding 系统")
    print("=" * 80)
    print(f"📁 FASTA: {args.fasta}")
    print(f"📊 VCF:   {args.vcf}")
    print(f"📋 GFF:   {args.gff}")
    print("=" * 80)

    output_dir = ensure_dir(args.output)
    ensure_dir(output_dir / "embeddings")
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "sequences")

    #ref_gene_pkl = output_dir / "ref_gene_embeddings.pkl"
    #ref_intergenic_pkl = output_dir / "intergenic_embeddings.pkl"


    # 0. 输入验证
    if not DataValidator.validate_inputs(args.fasta, args.vcf, args.gff):
        print("❌ 输入文件验证失败")
        return

    # 确保 VCF bgzip + index，并使用该路径
    vcf_path = DataValidator.ensure_bgzip_and_index(args.vcf)

    # 1. 加载 FASTA
    print("\n📥 步骤1: 加载 FASTA 序列")
    print("-" * 40)
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))
    genome_keys = list(genome.keys())
    print(f"✅ 加载 {len(genome)} 条染色体")

    # 2. 加载 GFF
    print("\n📥 步骤2: 加载 GFF 注释")
    print("-" * 40)
    gene_db = GeneDatabase(args.gff)
    all_genes = list(gene_db.genes.keys())
    print(f"🔹 GFF 中基因数量: {len(all_genes)}")

    # 3. 加载 VCF & 染色体映射
    print("\n📥 步骤3: 加载 VCF & 染色体命名标准化")
    print("-" * 40)
    pop_vcf = PopulationVCF(vcf_path)
    vcf_chroms = pop_vcf.get_chroms()
    gff_chroms = list({g['chrom'] for g in gene_db.genes.values()})

    chrom_mapping = DataValidator.normalize_chromosome_names(
        genome_keys, vcf_chroms, gff_chroms
    )

    # 4. 基因列表
    print("\n🔍 步骤4: 准备基因列表")
    print("-" * 40)

    if args.gene_list:
        print(f"📄 使用 gene_list 文件: {args.gene_list}")
        with open(args.gene_list) as f:
            filtered_genes = [line.strip() for line in f if line.strip() in gene_db.genes]
        print(f"📊 来自 gene_list 的基因数量: {len(filtered_genes)}")
    else:
        filtered_genes = list(gene_db.genes.keys())
        if args.max_genes:
            filtered_genes = filtered_genes[:args.max_genes]
            print(f"⚠️ 测试模式：仅处理前 {args.max_genes} 个基因")

    print(f"📊 最终处理基因数: {len(filtered_genes)}")

    # 5. 初始化 PlantCAD2
    print("\n🤖 步骤5: 初始化 PlantCAD2 模型")
    print("-" * 40)
    plantcad_embedder = PlantCAD2GeneEmbedder(
        model_name=args.model,
        device=args.device
    )
    '''
    locus_embedder = LocusEmbedderLongWindow(
        plantcad_embedder=plantcad_embedder,
        window_size=8192,
        stride=4096,
        pooling="mean"
    )
    '''
    locus_embedder = LocusEmbedderAdaptive(
        plantcad_embedder=plantcad_embedder,
        short_threshold=4096,
        window_size=4096,
        stride=2048,
        pooling="mean"
    )
    print("✅ PlantCAD2 模型已加载")
    # 6. 初始化 SequenceBuilder
    print("\n⚙️ 步骤6: 初始化 SequenceBuilder")
    print("-" * 40)
    seq_builder = SequenceBuilder(genome, gene_db, pop_vcf, chrom_mapping)
    print("✅ 序列构建器已准备")

    # 7. 构建 intergenic 区
    print("\n🧱 步骤7: 构建 intergenic regions")
    print("-" * 40)
    intergenic_regions = build_intergenic_regions(gene_db, genome, chrom_mapping)

    suffix = f".{selected_chrom}" if selected_chrom else ""
    ref_gene_pkl = output_dir / f"ref_gene_embeddings{suffix}.pkl"
    ref_intergenic_pkl = output_dir / f"intergenic_embeddings{suffix}.pkl"

    if selected_chrom:
        # 1) 过滤 genes
        filtered_genes = [
            gid for gid in filtered_genes
            if seq_builder._get_actual_chromosome(gene_db.get_gene_info(gid)['chrom']) == selected_chrom
        ]

        # 2) 过滤 intergenic
        intergenic_regions = [
            r for r in intergenic_regions
            if r['chrom_fasta'] == selected_chrom
        ]

        print(f"🧩 Chromosome mode = {selected_chrom}: genes={len(filtered_genes)}, intergenic={len(intergenic_regions)}")

    # 8. 生成参考基因 + 参考间区 embedding
    print("\n🚀 步骤8: 生成 reference gene & intergenic embedding")
    print("-" * 40)
    '''
    ref_gene_embeddings = {}
    failed_genes = []

    for gid in tqdm(filtered_genes, desc="Embedding reference genes"):
        try:
            seq = seq_builder.build_reference_sequence(gid, flank=args.flank)
            if not seq:
                failed_genes.append((gid, "no_sequence"))
                continue

            clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
            stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
            if stats['length'] < 50 or stats['n_content'] > 50:
                failed_genes.append((gid, "low_quality"))
                continue

            emb = locus_embedder.embed_locus(clean_seq)
            ref_gene_embeddings[gid] = emb
        except Exception as e:
            print(f"❌ 基因 {gid} 处理失败: {e}")
            failed_genes.append((gid, str(e)))
            continue

    ref_pkl = output_dir / "ref_gene_embeddings.pkl"
    with open(ref_pkl, "wb") as f:
        pickle.dump(ref_gene_embeddings, f)
    print(f"💾 参考基因 embedding 已保存到: {ref_pkl}")
    print(f"✅ 成功基因: {len(ref_gene_embeddings)}, 失败: {len(failed_genes)}")

    # 参考 intergenic
    intergenic_embeddings = {}
    for reg in tqdm(intergenic_regions, desc="Embedding reference intergenic"):
        chrom_fa = reg['chrom_fasta']
        start = reg['start']
        end = reg['end']

        try:
            seq = str(genome[chrom_fa].seq[start - 1:end]).upper()
            clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
            stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
            if stats['length'] < 50 or stats['n_content'] > 50:
                continue

            emb = locus_embedder.embed_locus(clean_seq)
            intergenic_embeddings[reg['id']] = {
                'chrom': chrom_fa,
                'start': start,
                'end': end,
                'embedding': emb
            }
        except Exception as e:
            print(f"❌ 间区 {reg['id']} 处理失败: {e}")
            continue

    intergenic_pkl = output_dir / "intergenic_embeddings.pkl"
    with open(intergenic_pkl, "wb") as f:
        pickle.dump(intergenic_embeddings, f)
    print(f"💾 参考 intergenic embedding 已保存到: {intergenic_pkl}")
    print(f"✅ 间区数量: {len(intergenic_embeddings)}")
    '''

    # ===============================
    # Step 8: Reference embedding
    # ===============================
    ref_gene_embeddings = {}
    intergenic_embeddings = {}

    if args.mode in ['reference', 'all']:
        print("\n🚀 Step 8: 生成 reference gene & intergenic embedding")

        # ---- reference gene ----
        failed_genes = []
        for gid in tqdm(filtered_genes, desc="Embedding reference genes"):
            try:
                seq = seq_builder.build_reference_sequence(gid, flank=args.flank)
                if not seq:
                    continue

                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue

                emb = locus_embedder.embed_locus(clean_seq)
                ref_gene_embeddings[gid] = emb

            except Exception as e:
                failed_genes.append((gid, str(e)))

        with open(ref_gene_pkl, "wb") as f:
            pickle.dump(ref_gene_embeddings, f)

        print(f"💾 reference gene embedding saved: {ref_gene_pkl}")

        # ---- reference intergenic ----
        for reg in tqdm(intergenic_regions, desc="Embedding reference intergenic"):
            chrom_fa = reg['chrom_fasta']
            start = reg['start']
            end = reg['end']

            try:
                seq = str(genome[chrom_fa].seq[start - 1:end]).upper()
                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue

                emb = locus_embedder.embed_locus(clean_seq)
                intergenic_embeddings[reg['id']] = {
                    'chrom': chrom_fa,
                    'start': start,
                    'end': end,
                    'embedding': emb
                }

            except Exception:
                continue

        with open(ref_intergenic_pkl, "wb") as f:
            pickle.dump(intergenic_embeddings, f)

        print(f"💾 reference intergenic embedding saved: {ref_intergenic_pkl}")


    # ===============================
    # Step 8 完成后
    # ===============================
    if args.mode == 'reference':
        print("\n🎉 Reference embedding 完成，程序退出（mode=reference）")
        return

    # ===============================
    # Step 9: Sample embedding
    # ===============================
    if args.mode in ['sample', 'all']:

        if not ref_gene_pkl.exists() or not ref_intergenic_pkl.exists():
            raise RuntimeError(
                "❌ reference embedding 不存在，请先运行 --mode reference"
            )

        print("\n📥 加载 reference embeddings ...")
        with open(ref_gene_pkl, "rb") as f:
            ref_gene_embeddings = pickle.load(f)

        with open(ref_intergenic_pkl, "rb") as f:
            intergenic_embeddings = pickle.load(f)

        print(
            f"✅ reference loaded: "
            f"{len(ref_gene_embeddings)} genes, "
            f"{len(intergenic_embeddings)} intergenic"
        )

    ##############################################
    ##############################################
    ##############################################
    
    # 9. 生成每个样本的 mutant embedding（基因 + 间区）
    print("\n🚀 步骤9: 生成每个样本的 mutant embedding（gene + intergenic, SNP only）")
    print("🔒 reference embedding 以只读模式加载（sample-only）")
    print("-" * 40)

    # 样本列表
    if args.samples:
        target_samples = [s for s in pop_vcf.samples if s in args.samples]
        print(f"👥 使用样本子集: {len(target_samples)}/{len(pop_vcf.samples)}")
    else:
        target_samples = pop_vcf.samples
        print(f"👥 全部样本数: {len(target_samples)}")

    for sample in target_samples:
        #sample_file = output_dir / f"sample_{sample}_embeddings.pkl"
        sample_file = output_dir / f"sample_{sample}{suffix}_embeddings.pkl"

        if sample_file.exists():
            print(f"⏭️  样本 {sample} 已存在结果，跳过（可作为 resume）")
            continue

        print(f"\n🧬 样本 {sample}: 生成 mutant embedding ...")
        
        snp_gene_count = 0
        snp_intergenic_count = 0

        t0 = time.perf_counter()

        print(f"🧠 构建 SNP index（样本 {sample}）...")
        #snp_index = SNPIndexBuilder(pop_vcf, seq_builder).build_for_sample(sample)
        
        chrom_len = len(genome[selected_chrom])

        snp_index = SNPIndexBuilder(
            pop_vcf, seq_builder
        ).build_for_sample_chrom(
            sample_name=sample,
            fasta_chrom=selected_chrom,
            chrom_len=chrom_len
        )

        print(f"   SNP 总数: {sum(len(v) for v in snp_index.values())}")


        sample_gene_embs = {}
        sample_intergenic_embs = {}

        '''
        # 基因
        for gid in tqdm(filtered_genes, desc=f"{sample} genes", leave=False):
            try:
                seq = seq_builder.build_sample_gene_sequence(gid, sample, flank=args.flank)
                if not seq:
                    continue
                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue
                emb = locus_embedder.embed_locus(clean_seq)
                sample_gene_embs[gid] = emb
            except Exception as e:
                print(f"❌ 样本 {sample}, 基因 {gid} 失败: {e}")
                continue
        '''

        # 基因
        for gid in tqdm(filtered_genes, desc=f"{sample} genes", leave=False):
            try:
                info = gene_db.get_gene_info(gid)
                if not info:
                    continue

                fasta_chrom = seq_builder._get_actual_chromosome(info['chrom'])
                start = max(1, info['start'] - args.flank)
                end = min(len(genome[fasta_chrom]), info['end'] + args.flank)

                # ===== SNP-aware 判断 =====
                has_snp = region_has_snp_fast(
                snp_index=snp_index,
                chrom=fasta_chrom,
                start=start,
                end=end)

                if has_snp:
                    snp_gene_count += 1

                # ===== 没有 SNP：直接复用 reference =====
                if not has_snp:
                    if gid in ref_gene_embeddings:
                        sample_gene_embs[gid] = ref_gene_embeddings[gid]
                    continue   # ←←← 关键！直接跳到下一个 gene

                # ===== 有 SNP：才真正构建 mutant + embed =====
                seq = seq_builder.build_sample_gene_sequence(gid, sample, flank=args.flank)
                if not seq:
                    continue

                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue

                emb = locus_embedder.embed_locus(clean_seq)
                sample_gene_embs[gid] = emb

            except Exception as e:
                print(f"❌ 样本 {sample}, 基因 {gid} 失败: {e}")
                continue

        '''                    
        # 间区
        for reg in tqdm(intergenic_regions, desc=f"{sample} intergenic", leave=False):
            chrom_fa = reg['chrom_fasta']
            start = reg['start']
            end = reg['end']
            rid = reg['id']

            try:
                seq = seq_builder.build_sample_locus_sequence(
                    fasta_chrom=chrom_fa,
                    start=start,
                    end=end,
                    sample_name=sample,
                    strand=None
                )
                if not seq:
                    continue
                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue

                emb = locus_embedder.embed_locus(clean_seq)
                sample_intergenic_embs[rid] = {
                    'chrom': chrom_fa,
                    'start': start,
                    'end': end,
                    'embedding': emb
                }
            except Exception as e:
                print(f"❌ 样本 {sample}, 间区 {rid} 失败: {e}")
                continue
        '''

        # 间区
        for reg in tqdm(intergenic_regions, desc=f"{sample} intergenic", leave=False):
            chrom_fa = reg['chrom_fasta']
            start = reg['start']
            end = reg['end']
            rid = reg['id']

            try:
                # ===== SNP-aware 判断 =====
                has_snp = region_has_snp_fast(
                snp_index=snp_index,
                chrom=chrom_fa,
                start=start,
                end=end
            )

                # ===== 没有 SNP：直接复用 reference intergenic =====
                if has_snp:
                    snp_intergenic_count += 1
                else:
                    # 没有 SNP：直接复用 reference
                    if rid in intergenic_embeddings:
                        sample_intergenic_embs[rid] = intergenic_embeddings[rid]
                    continue   # ← 不管有没有 reference，都不再算 mutant

                # ===== 有 SNP：才重新构建 + embed =====
                seq = seq_builder.build_sample_locus_sequence(
                    fasta_chrom=chrom_fa,
                    start=start,
                    end=end,
                    sample_name=sample,
                    strand=None
                )
                if not seq:
                    continue

                clean_seq, _ = SequenceQualityChecker.validate_dna_sequence(seq)
                stats = SequenceQualityChecker.get_sequence_stats(clean_seq)
                if stats['length'] < 50 or stats['n_content'] > 50:
                    continue
                emb = locus_embedder.embed_locus(clean_seq)
                sample_intergenic_embs[rid] = {
                    'chrom': chrom_fa,
                    'start': start,
                    'end': end,
                    'embedding': emb
                }

            except Exception as e:
                print(f"❌ 样本 {sample}, 间区 {rid} 失败: {e}")
                continue

        sample_data = {
            'sample': sample,
            'gene_embeddings': sample_gene_embs,
            'intergenic_embeddings': sample_intergenic_embs
        }

        with open(sample_file, "wb") as f:
            pickle.dump(sample_data, f)

        t1 = time.perf_counter()
        elapsed_min = (t1 - t0) / 60

        print(f"💾 样本 {sample} 的 mutant embedding 已保存到: {sample_file}")
        print(f"   基因数: {len(sample_gene_embs)}, 间区数: {len(sample_intergenic_embs)}")
        print(f"⏱️  样本 {sample} 用时: {elapsed_min:.2f} 分钟")


        total_genes = len(filtered_genes)
        total_intergenic = len(intergenic_regions)

        print(f"📊 SNP-aware 统计（样本 {sample}）:")
        print(f"  SNP genes:        {snp_gene_count} / {total_genes} "
            f"({snp_gene_count / total_genes * 100:.2f}%)")
        print(f"  SNP intergenic:   {snp_intergenic_count} / {total_intergenic} "
            f"({snp_intergenic_count / total_intergenic * 100:.2f}%)")
        print(f"  ⏱️ mutant embedding 用时: {elapsed_min:.2f} 分钟")


        gc.collect()

    print("\n" + "=" * 80)
    print("🎉 全流程完成!")
    print("=" * 80)
    print(f"📂 输出目录: {args.output}")
    print(f"📁 总文件大小: {get_directory_size(args.output):.2f} GB")


if __name__ == "__main__":
    main()
