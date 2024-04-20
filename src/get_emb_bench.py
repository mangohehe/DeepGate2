from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from progress.bar import Bar
import random
import time
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt

import utils.circuit_utils as circuit_utils
from config import get_parse_args, update_dir
from utils.random_seed import set_seed
from detectors.detector_factory import detector_factory
from datasets.load_data import parse_pyg_mlpgate

BENCH_DIR = "../dataset/itc99/"
TMP_DIR = "../tmp"
EMB_DIR = "../emb"
BENCH_NAMELIST = []
gate_to_index = {'INPUT': 0, 'AND': 1, 'NOT': 2}

def save_emb(emb, prob, path):
    f = open(path, 'w')
    f.write('{} {}\n'.format(len(emb), len(emb[0])))
    for i in range(len(emb)):
        for j in range(len(emb[i])):
            f.write('{:.6f} '.format(float(emb[i][j])))
        f.write('\n')
    for i in range(len(prob)):
        f.write('{:.6f}\n'.format(float(prob[i])))
    f.close()

def gen_graph(args, x_data, edge_index):
    x_data = np.array(x_data)
    edge_index = np.array(edge_index)
    tt_dis = []
    min_tt_dis = []
    tt_pair_index = []
    prob = [0] * len(x_data)
    rc_pair_index = [[0, 1]]
    is_rc = []
    g = parse_pyg_mlpgate(
        x_data, edge_index, tt_dis, min_tt_dis, tt_pair_index, prob, rc_pair_index, is_rc, 
        args.use_edge_attr, args.reconv_skip_connection, args.no_node_cop,
        args.node_reconv, args.un_directed, args.num_gate_types,
        args.dim_edge_feature, args.logic_implication, args.mask
    )
    return g

##################################################################
# API
##################################################################
def get_emb(exp_id, bench_filepath, emb_filepath, arch='mlpgnn', aggr='tfmlp', display=1):
    args = get_parse_args()
    args = update_dir(args, exp_id)
    args.arch = arch
    args.aggr_function = aggr
    args.batch_size = 1
    args.resume = True

    detector = detector_factory['base'](args)
    bench_name = bench_filepath.split('/')[-1].split('.')[0]
    x_data, edge_index, fanin_list, fanout_list, level_list = circuit_utils.parse_bench(bench_filepath, gate_to_index)
    if len(x_data) == 0:
        return 0
    if display:
        print('Parse AIG: ', bench_filepath)
    
    # Generate graph 
    g = gen_graph(args, x_data, edge_index)
    g.to(args.device)

    # Model 
    start_time = time.time()
    res = detector.run(g)
    end_time = time.time()
    hs, hf, prob, is_rc = res['results']
    if display:
        print("Circuit: {}, Size: {:}, Time: {:.2f} s".format(bench_name, len(x_data), end_time-start_time))
        print()

    # Save emb
    save_emb(hf.detach().cpu().numpy(), prob.detach().cpu().numpy(), emb_filepath)
    return 1
##################################################################

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    detector = detector_factory['base'](args)

    # Check if the data_dir is a directory or a specific file
    if os.path.isdir(args.data_dir):
        # It's a directory, process all .bench files in it
        bench_files = glob.glob(os.path.join(args.data_dir, '*.bench'))
    elif os.path.isfile(args.data_dir) and args.data_dir.endswith('.bench'):
        # It's a specific file
        bench_files = [args.data_dir]
    else:
        print("Error: data_dir is neither a directory nor a .bench file")
        return

    for filepath in bench_files:
        bench_name = os.path.basename(filepath).split('.')[0]
        print(f'[INFO] Read bench from bench file name: {filepath}')
        args.gate_to_index = gate_to_index
        print("Supported gates currently are:", list(args.gate_to_index.keys()))

        try:
            x_data, edge_index, fanin_list, fanout_list, level_list = circuit_utils.parse_bench(filepath, args.gate_to_index)
            if len(x_data) == 0:
                print(f"No data found in {bench_name}. Skipping...")
                continue

            print(f'Parsed AIG: {filepath}')

            # Generate graph 
            g = gen_graph(args, x_data, edge_index)
            g.to(args.device)

            # Model 
            start_time = time.time()
            res = detector.run(g)
            end_time = time.time()
            hs, hf, prob, is_rc = res['results']
            print(f"Circuit: {bench_name}, Size: {len(x_data)}, Time: {end_time - start_time:.2f} s")

            # Save emb
            emb_path = os.path.join(EMB_DIR, bench_name + '.txt')
            save_emb(hf.detach().cpu().numpy(), prob.detach().cpu().numpy(), emb_path)
        except Exception as e:
            print(f"Failed to process {bench_name}: {e}")
            
if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    test(args)

