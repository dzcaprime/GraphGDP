"""Based on https://github.com/ethanfetaya/NRI"""

"""Based on https://github.com/loeweX/AmortizedCausalDiscovery"""
"""Based on https://github.com/tailintalent/causal"""

import numpy as np
import time
import argparse
import networkx as nx
import kuramoto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="BA")
    parser.add_argument("--num-nodes", type=int, default=10, help="Number of nodes in the simulation.")
    parser.add_argument("--p", type=float, default=0.1, help="Connection/add connection probability In ER/NWS")
    parser.add_argument("--k", type=int, default=2, help="Inital node degree in BA/NWS")
    parser.add_argument("--num_all", type=int, default=10000, help="Number of total simulations to generate.")
    parser.add_argument("--exp_num", type=int, default=1, help="Number of repeated experiments")
    parser.add_argument("--length", type=int, default=51000, help="Length of trajectory.")
    parser.add_argument("--sample_freq", type=int, default=100, help="How often to sample the trajectory.")
    parser.add_argument(
        "--interaction_strength", type=int, default=1, help="Strength of Interactions between particles"
    )

    args = parser.parse_args()
    return args


def generate_edges(num_nodes, p, k, exp_id=None):
    graph_type = np.random.choice(["ER", "NWS", "BA"])
    n = num_nodes
    exp_id = np.random.randint(0, 10000) if exp_id is None else exp_id
    np.random.seed(exp_id)
    if graph_type in "ER":
        G = nx.erdos_renyi_graph(n, p, seed=exp_id)
    elif graph_type in "NWS":
        G = nx.newman_watts_strogatz_graph(n, k, p, seed=exp_id)
    elif graph_type in "BA":
        G = nx.barabasi_albert_graph(n, k, seed=exp_id)
    A = nx.to_numpy_array(G)
    return A


def generate_dataset(num_nodes, num_sims, length, sample_freq):
    num_sims = num_sims
    num_timesteps = int((length / float(sample_freq)) - 1)

    t0, t1, dt = 0, int((length / float(sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)

    sim_data_all = []
    edges_all = []
    edges = generate_edges(num_nodes, args.p, args.k)
    for i in range(num_sims):
        # Sample edges

        print(f"Simulating training trajectory:{i+1:3d}  /{num_sims:3d}")
        t = time.time()

        sim_data = kuramoto.simulate_kuramoto(edges, args.num_nodes, num_timesteps, T, dt)

        sim_data_all.append(sim_data)
        edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
    data_all = np.array(sim_data_all, dtype=np.float32)
    edges_all = np.stack(edges_all)
    return data_all, edges_all


if __name__ == "__main__":
    args = parse_args()
    suffix = "_Kuramoto"
    suffix += str(args.num_nodes)
    assert args.graph in {"ER", "NWS", "BA"}, "Unknown Graph Type"
    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k
        np.random.seed(exp_id)
        print("Generating data for experiment {}".format(exp_id))
        print("Generating {} simulations".format(args.num_all))
        data_all, edges_all = generate_dataset(n, args.num_all, args.length, args.sample_freq)
        # Reshape to [num_sims, num_timesteps, num_nodes, variables]
        data_all = data_all.transpose(0, 2, 1, 3)  # [num_sims, num_nodes, num_timesteps]
        print("Final data shape: ", data_all.shape)
        print("Final edges shape: ", edges_all.shape)
        # output data has shape [batch, time, nodes, variables]
        # save feat.npy and edges.npy
        np.save("feat" + suffix + ".npy", data_all)
        np.save("edges" + suffix + ".npy", edges_all)

        print("Experiment {exp_id} finished, saved to feat{suffix}.npy and edges{suffix}.npy".format(exp_id=exp_id, suffix=suffix))
