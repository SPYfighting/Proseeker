#!/usr/bin/env python3
"""
FoldX Stability Filtering Tool (versionB)

Function:
    Use FoldX physical energy function to evaluate thermodynamic stability of protein mutants,
    filter out predicted unstable candidate sequences (high DDG).

Dependencies:
    - FoldX 5.0+: http://foldxsuite.crg.eu/
    - Users need to download and install FoldX executable separately
    - FoldX requires license (free for academic use)

Usage example:
    python utils/filter_foldx_stability.py \
        --input data/candidates.csv \
        --pdb data/wildtype.pdb \
        --output data/candidates_stable.csv \
        --foldx_path /path/to/foldx \
        --ddg_threshold 5.0
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))


def check_foldx_installation(foldx_path: str) -> bool:
    if not os.path.exists(foldx_path):
        return False
    
    try:
        result = subprocess.run(
            [foldx_path, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def extract_sequence_alignment(parent_seq: str, child_seq: str) -> list[tuple[int, str, str]]:
    if len(parent_seq) != len(child_seq):
        raise ValueError("Parent and child sequence lengths are inconsistent")
    
    mutations = []
    for i, (p_aa, c_aa) in enumerate(zip(parent_seq, child_seq)):
        if p_aa != c_aa:
            mutations.append((i+1, p_aa, c_aa))
    
    return mutations


def create_foldx_mutation_file(mutations: list[tuple[int, str, str]], 
                               chain_id: str, output_path: str):
    """
    Create FoldX mutation input file (individual_list format)
    
    Args:
        mutations: [(position, original amino acid, new amino acid), ...]
        chain_id: PDB chain ID
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for pos, old_aa, new_aa in mutations:
            mutation_str = f"{old_aa}{chain_id}{pos}{new_aa}"
            f.write(mutation_str + ';\n')


def run_foldx_buildmodel(foldx_path: str, pdb_path: str, mutation_file: str, 
                         work_dir: str, chain_id: str = 'A') -> float | None:

    pdb_basename = os.path.basename(pdb_path)
    work_pdb = os.path.join(work_dir, pdb_basename)
    shutil.copy(pdb_path, work_pdb)
    
    work_mut_file = os.path.join(work_dir, 'individual_list.txt')
    shutil.copy(mutation_file, work_mut_file)
    
    command = [
        foldx_path,
        '--command=BuildModel',
        f'--pdb={pdb_basename}',
        '--mutant-file=individual_list.txt',
        '--numberOfRuns=3',
    ]
    
    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"Warning: FoldX execution failed: {result.stderr[:200]}")
            return None
        
        output_file = os.path.join(work_dir, f'Average_{pdb_basename.replace(".pdb", "")}_BuildModel.fxout')
        
        if not os.path.exists(output_file):
            possible_outputs = [f for f in os.listdir(work_dir) if 'BuildModel' in f and '.fxout' in f]
            if possible_outputs:
                output_file = os.path.join(work_dir, possible_outputs[0])
            else:
                return None
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                parts = lines[1].strip().split()
                try:
                    ddg = float(parts[1])  
                    return ddg
                except (ValueError, IndexError):
                    return None
        
        return None
        
    except subprocess.TimeoutExpired:
        print("Warning: FoldX execution timeout")
        return None
    except Exception as e:
        print(f"Warning: FoldX execution error: {e}")
        return None


def calculate_foldx_ddg_batch(foldx_path: str, pdb_path: str, 
                               parent_seq: str, child_sequences: list[str],
                               chain_id: str = 'A') -> list[float | None]:
    """
    Batch calculate FoldX DDG for multiple mutants
    """
    ddg_values = []
    
    with tempfile.TemporaryDirectory() as work_dir:
        for i, child_seq in enumerate(child_sequences):
            print(f"  Processing {i+1}/{len(child_sequences)}...", end='\r')
            
            try:
                mutations = extract_sequence_alignment(parent_seq, child_seq)
                
                if len(mutations) == 0:
                    ddg_values.append(0.0)
                    continue
                
                mut_file = os.path.join(work_dir, f'mut_{i}.txt')
                create_foldx_mutation_file(mutations, chain_id, mut_file)
                
                ddg = run_foldx_buildmodel(foldx_path, pdb_path, mut_file, work_dir, chain_id)
                ddg_values.append(ddg)
                
            except Exception as e:
                print(f"\nWarning: Sequence {i+1} processing failed: {e}")
                ddg_values.append(None)
    
    print()
    return ddg_values


def simulate_foldx_ddg(parent_seq: str, child_seq: str) -> float:
    """
    Simulate FoldX DDG calculation
    """
    num_mutations = sum(1 for p, c in zip(parent_seq, child_seq) if p != c)
    
    base_ddg = num_mutations * 1.5
    noise = np.random.randn() * 0.5
    
    return base_ddg + noise


def main():
    parser = argparse.ArgumentParser(
        description="Filter unstable mutation candidate sequences using FoldX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input candidate sequence CSV file')
    parser.add_argument('--pdb', type=str, required=True, help='Wild-type PDB file')
    parser.add_argument('--output', type=str, required=True, help='Output filtered CSV file')
    parser.add_argument('--foldx_path', type=str, default='foldx',
                       help='FoldX executable path (default assumes in PATH)')
    parser.add_argument('--ddg_threshold', type=float, default=5.0,
                       help='DDG threshold (kcal/mol), sequences exceeding this value will be filtered')
    parser.add_argument('--chain', type=str, default='A', help='PDB chain ID')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulation mode (do not call real FoldX)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FoldX Stability Filtering Tool (versionB)")
    print("="*70)
    print(f"Input file: {args.input}")
    print(f"PDB file: {args.pdb}")
    print(f"Output file: {args.output}")
    print(f"DDG threshold: {args.ddg_threshold} kcal/mol")
    print("="*70 + "\n")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if not os.path.exists(args.pdb):
        raise FileNotFoundError(f"PDB file does not exist: {args.pdb}")
    
    use_foldx = False
    if not args.simulate:
        if check_foldx_installation(args.foldx_path):
            print("FoldX detected")
            use_foldx = True
        else:
            print("Warning: FoldX not installed or path incorrect")
            print("  Using simulation mode instead")
            print("  Download FoldX: http://foldxsuite.crg.eu/")
            use_foldx = False
    
    print("\nReading candidate sequences...")
    df = pd.read_csv(args.input)
    print(f"  Total {len(df)} candidate sequences")
    
    if 'parent' not in df.columns or 'child' not in df.columns:
        raise ValueError("CSV file must contain 'parent' and 'child' columns")
    
    print(f"\nCalculating FoldX DDG{' (simulation mode)' if not use_foldx else ''}...")
    
    ddg_values = []
    
    if use_foldx:
        for parent_seq, group in df.groupby('parent'):
            child_seqs = group['child'].tolist()
            print(f"  Parent sequence {parent_seq[:20]}... ({len(child_seqs)} child sequences)")
            
            batch_ddg = calculate_foldx_ddg_batch(
                args.foldx_path, args.pdb, parent_seq, child_seqs, args.chain
            )
            ddg_values.extend(batch_ddg)
    else:
        for _, row in df.iterrows():
            ddg = simulate_foldx_ddg(row['parent'], row['child'])
            ddg_values.append(ddg)
        print(f"  Simulated {len(ddg_values)} DDG values")
    
    df['foldx_ddg'] = ddg_values
    
    valid_ddg = [d for d in ddg_values if d is not None]
    print(f"\nDDG Statistics:")
    print(f"  Valid calculations: {len(valid_ddg)}/{len(ddg_values)}")
    if valid_ddg:
        print(f"  Average DDG: {np.mean(valid_ddg):.2f} ± {np.std(valid_ddg):.2f} kcal/mol")
        print(f"  Range: [{np.min(valid_ddg):.2f}, {np.max(valid_ddg):.2f}]")
    
    print(f"\nApplying stability filter (DDG < {args.ddg_threshold} kcal/mol)...")
    df_filtered = df[
        (df['foldx_ddg'].notna()) & 
        (df['foldx_ddg'] < args.ddg_threshold)
    ].copy()
    
    removed_count = len(df) - len(df_filtered)
    print(f"  Retained: {len(df_filtered)} sequences")
    print(f"  Filtered: {removed_count} sequences ({removed_count/len(df)*100:.1f}%)")
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_filtered.to_csv(args.output, index=False)
    
    print(f"\nFiltered sequences saved to: {args.output}")
    
    full_output = args.output.replace('.csv', '_full_ddg.csv')
    df.to_csv(full_output, index=False)
    print(f"  Full results (with DDG): {full_output}")
    
    print("\nNext steps:")
    print("  Use filtered sequences for model prediction or experimental validation")
    print()


if __name__ == '__main__':
    main()

