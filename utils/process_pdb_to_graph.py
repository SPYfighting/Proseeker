#!/usr/bin/env python3
"""
PDB Structure to Graph Preprocessing Script (versionB - Substrate Aware)
Usage example:
    python utils/process_pdb_to_graph.py \
        --pdb data/complex.pdb \
        --output data/wt_graph.pt \
        --ligand_name DTP \
        --pocket_threshold 6.0
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))


def get_structure_and_chain(pdb_path: str, chain_id: str | None = None):
    """Helper function: Load structure and get specified chain"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    
    if chain_id:
        if chain_id not in [chain.id for chain in model]:
            raise ValueError(f"Chain {chain_id} does not exist in PDB file")
        chain = model[chain_id]
    else:
        chain = list(model.get_chains())[0]
    
    return structure, model, chain


def extract_ca_coordinates(chain) -> tuple[np.ndarray, list[int]]:
    """
    Extract Cα atom coordinates from PDB chain object
    """
    coords = []
    residue_ids = []
    
    print(f"Processing chain: {chain.id}")
    
    for residue in chain:
        if is_aa(residue, standard=True):
            if 'CA' in residue:
                ca_atom = residue['CA']
                coords.append(ca_atom.get_coord())
                residue_ids.append(residue.id[1])
    
    if len(coords) == 0:
        raise ValueError("No Cα atoms found in PDB file")
    
    coords = np.array(coords)
    print(f"Extracted {len(coords)} Cα atoms (corresponding to protein sequence length)")
    
    return coords, residue_ids


def get_binding_pocket_indices(model, target_chain, ligand_name: str, threshold: float = 5.0) -> list[int]:
    """
    Identify residue indices within threshold distance from substrate (0-based index)
    """
    print(f"Searching for substrate: {ligand_name} ...")
    
    ligand_atoms = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname == ligand_name:
                ligand_atoms.extend(residue.get_atoms())
            
    if not ligand_atoms:
        print(f"Warning: Substrate named {ligand_name} not found! Binding site information will not be used.")
        print("  Hint: Please check if the residue name (RESNAME) in the PDB file is correct.")
        return []

    print(f"  Found substrate atoms: {len(ligand_atoms)}")

    binding_indices = []
    current_idx = 0
    
    lig_coords = np.array([a.get_coord() for a in ligand_atoms])
    
    for residue in target_chain:
        if is_aa(residue, standard=True):
            if 'CA' in residue:
                res_atoms = [a.get_coord() for a in residue.get_atoms()]
                res_coords = np.array(res_atoms)
                
                diff = res_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dist = np.min(dists)
                
                if min_dist < threshold:
                    binding_indices.append(current_idx)
                
                current_idx += 1
                
    print(f"Identified substrate binding site residue indices: {binding_indices}")
    print(f"  ({len(binding_indices)} residues within {threshold}Å range)")
    return binding_indices


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate Cα-Cα Euclidean distance matrix between residues"""
    N = coords.shape[0]
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(delta**2, axis=-1))
    return distance_matrix


def build_graph_from_distances(distance_matrix: np.ndarray, 
                               threshold: float = 10.0, 
                               include_sequential: bool = True,
                               binding_indices: list[int] = None) -> torch.Tensor:
    """
    Build graph based on distance threshold and binding site information
    """
    N = distance_matrix.shape[0]
    edge_list = []
    
    binding_set = set(binding_indices) if binding_indices else set()
    
    geo_edges = 0
    seq_edges = 0
    pocket_edges = 0
    
    for i in range(N):
        for j in range(i+1, N):
            is_connected = False
            type_flag = ""
            
            # 1. Geometric distance connection
            if distance_matrix[i, j] < threshold:
                is_connected = True
                geo_edges += 1
                type_flag = "geo"
            
            # 2. Sequential neighbor connection
            elif include_sequential and abs(i - j) == 1:
                is_connected = True
                seq_edges += 1
                type_flag = "seq"
            
            elif (i in binding_set) and (j in binding_set):
                is_connected = True
                pocket_edges += 1
                type_flag = "pocket"
                
            if is_connected:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    if len(edge_list) == 0:
        print("Warning: No edges found, adding self-loop edges")
        edge_list = [[i, i] for i in range(N)]
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"Graph construction details:")
    print(f"  - Geometric distance edges (<{threshold}Å): {geo_edges}")
    print(f"  - Sequential neighbor completion: {seq_edges}")
    if binding_indices:
        print(f"  - Binding pocket enhanced edges: {pocket_edges}")
    print(f"  - Total edges: {edge_index.shape[1]}")
    
    return edge_index


def visualize_graph_statistics(distance_matrix: np.ndarray, edge_index: torch.Tensor, threshold: float):
    """Print graph statistics"""
    N = distance_matrix.shape[0]
    num_edges = edge_index.shape[1]
    degrees = torch.zeros(N, dtype=torch.long)
    for edge in edge_index.t():
        degrees[edge[0]] += 1
    
    print("\n" + "="*60)
    print("Graph Statistics")
    print("="*60)
    print(f"Number of nodes: {N}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {degrees.float().mean().item():.2f}")
    print(f"Density: {num_edges / (N * (N - 1)):.4f}")
    print("="*60 + "\n")


def save_graph(edge_index: torch.Tensor, output_path: str, metadata: dict | None = None):
    """Save graph structure to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dict = {
        'edge_index': edge_index,
        'num_nodes': edge_index.max().item() + 1,
    }
    if metadata:
        save_dict['metadata'] = metadata
    torch.save(save_dict, output_path)
    print(f"Graph structure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB file to graph structure (supports substrate binding site awareness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--pdb', type=str, required=True, help='Input PDB file path')
    parser.add_argument('--output', type=str, required=True, help='Output graph file path (.pt format)')
    parser.add_argument('--chain', type=str, default=None, help='Specify protein chain ID')
    parser.add_argument('--threshold', type=float, default=10.0, help='Cα-Cα distance threshold (Angstrom)')
    parser.add_argument('--no-sequential', action='store_true', help='Do not force connection of sequentially adjacent residues')
    
    parser.add_argument('--ligand_name', type=str, default=None,
                        help='Substrate residue name (e.g., DTP, 3ON), used to enhance binding site graph connections')
    parser.add_argument('--pocket_threshold', type=float, default=7.0,
                        help='Distance threshold for determining residues belonging to binding pocket (Angstrom)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PDB to Graph Conversion Tool (versionB - Enhanced)")
    print("="*60)
    print(f"Input PDB: {args.pdb}")
    if args.ligand_name:
        print(f"Substrate mode: Enabled (Ligand: {args.ligand_name}, Dist: {args.pocket_threshold}Å)")
    else:
        print(f"Substrate mode: Disabled (Pure geometric distance)")
    print("="*60 + "\n")
    
    if not os.path.exists(args.pdb):
        raise FileNotFoundError(f"PDB file does not exist: {args.pdb}")
    
    structure, model, chain = get_structure_and_chain(args.pdb, args.chain)
    
    coords, residue_ids = extract_ca_coordinates(chain)
    
    binding_indices = None
    if args.ligand_name:
        binding_indices = get_binding_pocket_indices(
            model, chain, args.ligand_name, args.pocket_threshold
        )
    
    print("Calculating distance matrix...")
    distance_matrix = compute_distance_matrix(coords)
    
    print(f"Building graph...")
    edge_index = build_graph_from_distances(
        distance_matrix, 
        threshold=args.threshold,
        include_sequential=not args.no_sequential,
        binding_indices=binding_indices
    )
    
    visualize_graph_statistics(distance_matrix, edge_index, args.threshold)
    
    metadata = {
        'pdb_file': args.pdb,
        'chain_id': chain.id,
        'threshold': args.threshold,
        'num_residues': len(coords),
        'residue_ids': residue_ids,
        'ligand_name': args.ligand_name,
        'binding_site_indices': binding_indices
    }
    save_graph(edge_index, args.output, metadata)
    
    print("Conversion completed!")
    if args.ligand_name and binding_indices:
        print(f"Hint: Encoded {len(binding_indices)} pocket residues and their relationships into the graph.")


if __name__ == '__main__':
    main()