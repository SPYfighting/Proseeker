#!/usr/bin/env python3
"""
ProteinMPNN Sequence Generation Tool (versionB)

Function:
    Based on fixed wild-type (WT) backbone structure, use ProteinMPNN model to generate
    high-quality combinatorial mutation sequences at specified positions to expand candidate library.

Dependencies:
    - ProteinMPNN: https://github.com/dauparas/ProteinMPNN
    - Requires pre-installed ProteinMPNN environment
    
Usage example:
    python utils/generate_mpnn_candidates.py \
        --pdb data/wildtype.pdb \
        --positions 10,25,48,72 \
        --output data/mpnn_candidates.csv \
        --num_samples 100
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))

try:
    PROTEINMPNN_AVAILABLE = False
    print("Warning: This is a template implementation of ProteinMPNN tool")
    print("  Please ensure ProteinMPNN is installed: https://github.com/dauparas/ProteinMPNN")
except ImportError:
    PROTEINMPNN_AVAILABLE = False


def extract_sequence_from_pdb(pdb_path: str, chain_id: str | None = None) -> tuple[str, list[int]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    
    if chain_id:
        chain = model[chain_id]
    else:
        chain = list(model.get_chains())[0]
        print(f"Using chain: {chain.id}")
    
    AA_3TO1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'MSE': 'M',
        'SEC': 'C',
        'PYL': 'O',
    }
    
    def three_to_one(resname: str) -> str:
        resname_upper = resname.upper().strip()
        return AA_3TO1.get(resname_upper, 'X')
    
    sequence = []
    residue_ids = []
    
    for residue in chain:
        if is_aa(residue, standard=True):
            try:
                aa = three_to_one(residue.get_resname())
                sequence.append(aa)
                residue_ids.append(residue.id[1])
            except (KeyError, ValueError):
                continue
    
    return ''.join(sequence), residue_ids


def run_proteinmpnn_design(pdb_path: str, designable_positions: list[int], 
                           num_samples: int = 100, temperature: float = 0.1) -> list[str]:
    """
    Run ProteinMPNN for sequence design (simulated implementation)
    
    Args:
        pdb_path: PDB file path
        designable_positions: Designable residue positions (0-based index)
        num_samples: 生成样本数量
        temperature: 采样温度
    
    返回:
        designed_sequences: 设计的序列列表
    """
    if not PROTEINMPNN_AVAILABLE:
        print("Warning: ProteinMPNN not installed, using simulated generation mode")
        return simulate_mpnn_design(pdb_path, designable_positions, num_samples)
    
    """
    from protein_mpnn import ProteinMPNN
    
    model = ProteinMPNN.load_model()
    structure = parse_PDB(pdb_path)
    
    design_mask = torch.zeros(len(structure))
    design_mask[designable_positions] = 1
    
    designed_seqs = model.design(
        structure=structure,
        design_mask=design_mask,
        num_samples=num_samples,
        temperature=temperature
    )
    
    return designed_seqs
    """
    
    raise NotImplementedError("Please install ProteinMPNN and implement this function")


def simulate_mpnn_design(pdb_path: str, designable_positions: list[int], 
                        num_samples: int = 100) -> list[str]:
    """
    Simulate ProteinMPNN design process (for demonstration)
    
    Should be replaced with real ProteinMPNN call in actual use
    """
    print("Using simulation mode to generate candidate sequences...")
    
    wt_sequence, residue_ids = extract_sequence_from_pdb(pdb_path)
    print(f"  Wild-type sequence length: {len(wt_sequence)}")
    print(f"  Number of designable positions: {len(designable_positions)}")
    
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    designed_sequences = []
    
    for _ in range(num_samples):
        seq_list = list(wt_sequence)
        
        for pos in designable_positions:
            if 0 <= pos < len(seq_list):
                original_aa = seq_list[pos]
                new_aa = np.random.choice([aa for aa in amino_acids if aa != original_aa])
                seq_list[pos] = new_aa
        
        designed_sequences.append(''.join(seq_list))
    
    designed_sequences = list(set(designed_sequences))
    print(f"  Generated {len(designed_sequences)} unique sequences")
    
    return designed_sequences


def save_candidates_to_csv(parent_seq: str, child_sequences: list[str], 
                           output_path: str, mutation_type: str = 'mpnn'):
    """
    Save candidate sequences to CSV file
    
    Args:
        parent_seq: Parent sequence (WT)
        child_sequences: List of child sequences
        output_path: Output file path
        mutation_type: Mutation type label
    """
    data = []
    
    for child_seq in child_sequences:
        num_mutations = sum(1 for p, c in zip(parent_seq, child_seq) if p != c)
        
        data.append({
            'parent': parent_seq,
            'child': child_seq,
            'mutation_type': mutation_type,
            'num_mutations': num_mutations,
        })
    
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df)} candidate sequences to: {output_path}")
    print(f"  Average mutations: {df['num_mutations'].mean():.1f}")
    print(f"  Mutation range: [{df['num_mutations'].min()}, {df['num_mutations'].max()}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate mutation sequences using ProteinMPNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--pdb', type=str, required=True, help='Wild-type PDB file path')
    parser.add_argument('--positions', type=str, required=True,
                       help='Designable residue positions, comma-separated (0-based index), e.g.: 10,25,48')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='MPNN sampling temperature')
    parser.add_argument('--chain', type=str, default=None, help='PDB chain ID (default first chain)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ProteinMPNN Candidate Sequence Generation Tool (versionB)")
    print("="*70)
    print(f"Input PDB: {args.pdb}")
    print(f"Output file: {args.output}")
    print(f"Number to generate: {args.num_samples}")
    print("="*70 + "\n")
    
    if not os.path.exists(args.pdb):
        raise FileNotFoundError(f"PDB file does not exist: {args.pdb}")
    
    try:
        designable_positions = [int(x.strip()) for x in args.positions.split(',')]
        print(f"Designable positions: {designable_positions}")
    except ValueError:
        raise ValueError("Position parameter format error, should be comma-separated integers, e.g.: 10,25,48")
    
    print("\nExtracting wild-type sequence...")
    wt_sequence, residue_ids = extract_sequence_from_pdb(args.pdb, args.chain)
    print(f"Sequence length: {len(wt_sequence)}")
    print(f"  Sequence preview: {wt_sequence[:50]}{'...' if len(wt_sequence) > 50 else ''}")
    
    invalid_positions = [p for p in designable_positions if p < 0 or p >= len(wt_sequence)]
    if invalid_positions:
        print(f"Warning: Following positions out of sequence range, will be ignored: {invalid_positions}")
        designable_positions = [p for p in designable_positions if 0 <= p < len(wt_sequence)]
    
    print(f"\nStarting ProteinMPNN design...")
    print(f"  Temperature: {args.temperature}")
    print(f"  Samples: {args.num_samples}")
    
    try:
        designed_sequences = run_proteinmpnn_design(
            pdb_path=args.pdb,
            designable_positions=designable_positions,
            num_samples=args.num_samples,
            temperature=args.temperature
        )
    except NotImplementedError:
        print("  Using simulation mode instead...")
        designed_sequences = simulate_mpnn_design(
            pdb_path=args.pdb,
            designable_positions=designable_positions,
            num_samples=args.num_samples
        )
    
    save_candidates_to_csv(wt_sequence, designed_sequences, args.output, mutation_type='mpnn')
    
    print("\nGeneration completed!")
    print("\nNext steps:")
    print("  1. Score candidate sequences using model:")
    print("     python pipeline/04_predict_with_uncertainty.py --candidates", args.output)
    print("  2. Optional: Filter stability using FoldX:")
    print("     python utils/filter_foldx_stability.py --input", args.output, "--pdb", args.pdb)
    print()


if __name__ == '__main__':
    main()

