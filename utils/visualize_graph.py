#!/usr/bin/env python3
"""
Protein Graph Structure Visualization Tool

Functions:
1. Load wt_graph.pt file
2. Extract sequence information from PDB file
3. Visualize graph structure with amino acid annotations on nodes

Usage example:
    python utils/visualize_graph.py \
        --graph data/wt_graph.pt \
        --pdb data/wt.pdb \
        --output graph_visualization.png
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import warnings

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
    """
    Convert three-letter amino acid code to one-letter code
    
    Args:
        resname: Three-letter amino acid code (e.g., 'ALA', 'GLY')
    
    Returns:
        One-letter code (e.g., 'A', 'G'), returns 'X' if unrecognized
    """
    resname_upper = resname.upper().strip()
    return AA_3TO1.get(resname_upper, 'X')

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
if ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(ROOT))


def extract_sequence_from_pdb(pdb_path: str, chain_id: str | None = None) -> tuple[str, list[int]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    
    if chain_id:
        if chain_id not in [chain.id for chain in model]:
            raise ValueError(f"Chain {chain_id} does not exist in PDB file")
        chain = model[chain_id]
    else:
        chain = list(model.get_chains())[0]
        print(f"Using chain: {chain.id}")
    
    sequence = []
    residue_ids = []
    
    for residue in chain:
        if is_aa(residue, standard=True):
            try:
                aa = three_to_one(residue.get_resname())
                sequence.append(aa)
                residue_ids.append(residue.id[1])
            except KeyError:
                continue
    
    return ''.join(sequence), residue_ids


def extract_ca_coordinates(pdb_path: str, chain_id: str | None = None) -> np.ndarray:
    """
    Extract Cα atom coordinates from PDB file (for 3D layout)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    
    if chain_id:
        chain = model[chain_id]
    else:
        chain = list(model.get_chains())[0]
    
    coords = []
    for residue in chain:
        if is_aa(residue, standard=True):
            if 'CA' in residue:
                ca_atom = residue['CA']
                coords.append(ca_atom.get_coord())
    
    return np.array(coords)


def load_graph(graph_path: str) -> tuple[torch.Tensor, dict]:
    graph_data = torch.load(graph_path, map_location='cpu')
    edge_index = graph_data['edge_index']
    metadata = graph_data.get('metadata', {})
    
    print(f"Loaded graph structure: {graph_path}")
    print(f"  Nodes: {graph_data['num_nodes']}")
    print(f"  Edges: {edge_index.shape[1]}")
    
    return edge_index, metadata


def visualize_graph_2d(edge_index: torch.Tensor, sequence: str, 
                      binding_indices: list[int] | None = None,
                      output_path: str = "graph_visualization.png",
                      layout: str = "spring",
                      node_size: int = 500,
                      font_size: int = 8,
                      figsize: tuple = (20, 16)):
    # 创建 NetworkX 图
    G = nx.Graph()
    
    num_nodes = edge_index.max().item() + 1
    for i in range(num_nodes):
        G.add_node(i)
    
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    print(f"Using {layout} layout algorithm...")
    if layout == "spring":
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    node_colors = []
    if binding_indices:
        binding_set = set(binding_indices)
        node_colors = ['red' if i in binding_set else 'lightblue' for i in range(num_nodes)]
    else:
        node_colors = ['lightblue'] * num_nodes
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_size, alpha=0.8)
    
    labels = {}
    for i in range(min(len(sequence), num_nodes)):
        labels[i] = f"{sequence[i]}\n{i+1}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, 
                           font_weight='bold', font_color='black')
    
    title = f"Protein Graph Structure Visualization\n"
    title += f"Nodes: {num_nodes}, Edges: {G.number_of_edges()}, Sequence length: {len(sequence)}"
    if binding_indices:
        title += f", Binding pocket residues: {len(binding_indices)}"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()


def visualize_graph_3d(edge_index: torch.Tensor, sequence: str,
                      pdb_path: str, chain_id: str | None = None,
                      binding_indices: list[int] | None = None,
                      output_path: str = "graph_visualization_3d.png",
                      node_size: int = 100,
                      font_size: int = 6,
                      figsize: tuple = (20, 16)):
    from mpl_toolkits.mplot3d import Axes3D
    
    # 提取 Cα 坐标
    coords = extract_ca_coordinates(pdb_path, chain_id)
    num_nodes = edge_index.max().item() + 1
    
    if len(coords) < num_nodes:
        print(f"⚠️  警告: PDB 坐标数 ({len(coords)}) < 节点数 ({num_nodes})")
        print("   将使用 2D 布局代替")
        return visualize_graph_2d(edge_index, sequence, binding_indices, output_path)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    x_coords = coords[:num_nodes, 0]
    y_coords = coords[:num_nodes, 1]
    z_coords = coords[:num_nodes, 2]
    
    # 节点颜色
    node_colors = []
    if binding_indices:
        binding_set = set(binding_indices)
        node_colors = ['red' if i in binding_set else 'lightblue' for i in range(num_nodes)]
    else:
        node_colors = ['lightblue'] * num_nodes
    
    # 绘制边
    edge_list = edge_index.t().tolist()
    for edge in edge_list:
        i, j = edge
        ax.plot([x_coords[i], x_coords[j]], 
                [y_coords[i], y_coords[j]], 
                [z_coords[i], z_coords[j]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    # 计算中心点（用于文本偏移）
    center_x = x_coords.mean()
    center_y = y_coords.mean()
    center_z = z_coords.mean()
    
    # 计算坐标范围，用于确定合适的偏移量
    coord_range = max(
        x_coords.max() - x_coords.min(),
        y_coords.max() - y_coords.min(),
        z_coords.max() - z_coords.min()
    )
    # 偏移量设为坐标范围的2-3%
    offset_scale = coord_range * 0.025
    
    # 绘制节点
    for i in range(num_nodes):
        ax.scatter(x_coords[i], y_coords[i], z_coords[i], 
                  c=node_colors[i], s=node_size, alpha=0.8)
        
        # 只在结合口袋残基显示标签
        if binding_indices and i in binding_indices and i < len(sequence):
            # 计算从中心到节点的方向向量
            dx = x_coords[i] - center_x
            dy = y_coords[i] - center_y
            dz = z_coords[i] - center_z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 归一化方向向量并应用偏移
            if dist > 0:
                offset_x = x_coords[i] + (dx / dist) * offset_scale
                offset_y = y_coords[i] + (dy / dist) * offset_scale
                offset_z = z_coords[i] + (dz / dist) * offset_scale
            else:
                offset_x = x_coords[i] + offset_scale
                offset_y = y_coords[i] + offset_scale
                offset_z = z_coords[i] + offset_scale
            
            # 添加标签（使用偏移位置）
            ax.text(offset_x, offset_y, offset_z, 
                   f"{sequence[i]}{i+1}", 
                   fontsize=font_size, fontweight='bold',
                   color='darkred',  # 深红色，更醒目
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='darkred', alpha=0.8, linewidth=1.5))
    
    # 设置标签
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    
    # 添加标题
    title = f"蛋白质图结构可视化 (3D)\n"
    title += f"节点数: {num_nodes}, 边数: {len(edge_list)}, 序列长度: {len(sequence)}"
    if binding_indices:
        title += f", 结合口袋残基: {len(binding_indices)}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 3D 图片已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化蛋白质图结构",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--graph', type=str, required=True, 
                       help='图文件路径 (.pt 格式)')
    parser.add_argument('--pdb', type=str, default=None,
                       help='PDB 文件路径（用于提取序列和3D坐标）')
    parser.add_argument('--output', type=str, default='graph_visualization.png',
                       help='输出图片路径')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai', 'spectral', '3d'],
                       help='布局算法（2D）或3D模式')
    parser.add_argument('--node_size', type=int, default=500,
                       help='节点大小')
    parser.add_argument('--font_size', type=int, default=8,
                       help='字体大小')
    parser.add_argument('--figsize', type=str, default='20,16',
                       help='图片大小 (宽,高)')
    
    args = parser.parse_args()
    
    # 解析图片大小
    figsize = tuple(map(int, args.figsize.split(',')))
    
    print("\n" + "="*60)
    print("🎨 蛋白质图结构可视化工具")
    print("="*60)
    print(f"图文件: {args.graph}")
    print(f"输出文件: {args.output}")
    print("="*60 + "\n")
    
    # 加载图
    edge_index, metadata = load_graph(args.graph)
    
    # 获取序列信息
    sequence = None
    pdb_path = args.pdb or metadata.get('pdb_file')
    chain_id = metadata.get('chain_id')
    
    if pdb_path and os.path.exists(pdb_path):
        print(f"📖 从 PDB 文件提取序列: {pdb_path}")
        sequence, residue_ids = extract_sequence_from_pdb(pdb_path, chain_id)
        print(f"✅ 序列长度: {len(sequence)}")
        print(f"   序列前20个残基: {sequence[:20]}")
    else:
        print("⚠️  警告: 未找到 PDB 文件，无法提取序列信息")
        print("   将使用节点索引作为标签")
        num_nodes = edge_index.max().item() + 1
        sequence = '?' * num_nodes
    
    # 获取结合口袋信息
    binding_indices = metadata.get('binding_site_indices')
    if binding_indices:
        print(f"🔐 结合口袋残基数: {len(binding_indices)}")
    
    # 可视化
    if args.layout == '3d':
        if not pdb_path or not os.path.exists(pdb_path):
            print("⚠️  警告: 3D 模式需要 PDB 文件，将使用 2D spring 布局")
            visualize_graph_2d(edge_index, sequence, binding_indices, 
                              args.output, 'spring', args.node_size, 
                              args.font_size, figsize)
        else:
            visualize_graph_3d(edge_index, sequence, pdb_path, chain_id,
                              binding_indices, args.output, args.node_size,
                              args.font_size, figsize)
    else:
        visualize_graph_2d(edge_index, sequence, binding_indices,
                          args.output, args.layout, args.node_size,
                          args.font_size, figsize)
    
    print("\n✅ 可视化完成！")


if __name__ == '__main__':
    main()

