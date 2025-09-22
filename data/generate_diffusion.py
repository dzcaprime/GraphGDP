import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import argparse
import pickle

parser = argparse.ArgumentParser('Generate diffusion equation data')
parser.add_argument('--graph', type=str, default='ER')
parser.add_argument('--num-nodes', type=int, default=10,
                    help='Number of nodes in the simulation.')
parser.add_argument('--p', type=float, default=0.1, 
                    help='Connection/add connection probability In ER/NWS')
parser.add_argument('--k', type=int, default=2, 
                    help='Inital node degree in BA/NWS')

parser.add_argument('--exp_num', type=int, default=1,
                    help='Number of repeated experiments')
parser.add_argument('--tr_num', type=int, default=100,
                    help='Number of train trajectories.')
parser.add_argument('--va_num', type=int, default=30,
                    help='Number of validation trajectories.')
parser.add_argument('--te_num', type=int, default=30,
                    help='Number of test trajectories.')

parser.add_argument('--steps', type=int, default=200, help='simulation times steps')
# 将 step_size 类型修正为 float，保持默认值与用法一致
parser.add_argument('--step_size', type=float, default=0.1, help = 'simulation step size')
parser.add_argument('--beta', type=float, default=1.0, help='diffusion constant')

# 新增：Kuramoto 风格保存参数
parser.add_argument('--save_kuramoto_style', action='store_true',
                    help='If set, save results like generate_kuramoto.py: '
                         'feat_Diffusion{N}.npy [B,T,N,1] and edges_Diffusion{N}.npy [B,N,N]. '
                         'Uses a single graph per experiment and stacks it B times.')
parser.add_argument('--num_all', type=int, default=10000,
                    help='Total number of trajectories (B) to generate when --save_kuramoto_style '
                         'is set. Default 10000 to match Kuramoto script.')
args = parser.parse_args()


def field(x, t):
    """
    Compute dx/dt = beta * L @ x.

    Why
    ---
    Keep diffusion dynamics intact; only I/O and saving paths are modified.
    L and beta are bound from outer scope to avoid passing through ODE solver.
    """
    dxdt = beta*L@x
    return dxdt  

# solve the ODE with an ODE solver
def simulation(t):
    """
    Run one diffusion simulation for a random initial condition.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape [T].

    Returns
    -------
    np.ndarray
        Trajectory with shape [T, N, 1].
    """
    x0 = np.random.random(size=n)
    x = odeint(field,x0,t)
    x = np.expand_dims(x, axis = -1) 
    return x

def plot_trajectory(x,t):
    for nid in range(n):
        plt.plot(t,x[:,nid])
        plt.xlabel('time')
        plt.ylabel('x(t)')
    plt.show()

if __name__ == '__main__':

    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'

    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k
        beta = args.beta
        np.random.seed(exp_id)
        # 明确等号判断，行为不变但更清晰
        if args.graph == 'ER':
            G = nx.erdos_renyi_graph(n,p,seed=exp_id)
        elif args.graph == 'NWS':
            G = nx.newman_watts_strogatz_graph(n,k,p,seed=exp_id)
        elif args.graph == 'BA':
            G = nx.barabasi_albert_graph(n,k,seed=exp_id)

        # get laplacian
        A = nx.to_numpy_array(G)
        D = A.sum(axis=1) + 1e-6
        D = 1 / np.sqrt(D)
        D = np.diag(D)
        L = - D@A@D

        t = np.linspace(0, (args.steps - 1)* args.step_size, num=args.steps)

        # 新增：当选择 Kuramoto 风格保存时，按同一张图生成并堆叠 B 条轨迹
        if args.save_kuramoto_style:
            B = int(args.num_all)
            sim_data_all = []
            for i in range(B):
                if (i % 100) == 0:
                    print(f"Simulating trajectory: {i+1:6d}/{B:6d}")
                sim_data_all.append(simulation(t))
            data_all = np.array(sim_data_all, dtype=np.float32)  # [B, T, N, 1]
            edges_all = np.repeat(A[None, ...], B, axis=0)      # [B, N, N]

            suffix = "_Diffusion" + str(n)
            print("Final data shape: ", data_all.shape)
            print("Final edges shape: ", edges_all.shape)
            np.save("feat" + suffix + ".npy", data_all)
            np.save("edges" + suffix + ".npy", edges_all)
            print("Experiment {exp_id} finished, saved to feat{suffix}.npy and edges{suffix}.npy"
                  .format(exp_id=exp_id, suffix=suffix))
            continue  # 不执行旧的 pickle 保存路径

        # 保持原有行为：生成 tr/va/te 并保存为 pickle（兼容旧用户）
        x_tr = np.zeros((args.tr_num,args.steps, n ,1))
        for i in range(args.tr_num):
            print(f'Simulating train trajectory: {i+1:3d}/{args.tr_num:3d}')
            x_tr[i] = simulation(t)
            #plot_trajectory(x_tr[i],t)

        x_va = np.zeros((args.va_num,args.steps,n,1))
        for i in range(args.va_num):
            print(f'Simulating  validation trajectory: {i+1:3d}/{args.va_num:3d}')
            x_va[i] = simulation(t)

        x_te = np.zeros((args.te_num,args.steps,n,1))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1:3d}/{args.te_num:3d}')
            x_te[i] = simulation(t)

        #A = nx.to_numpy_array(G)

        x_tr = x_tr.astype(np.float16)
        x_va = x_va.astype(np.float16)
        x_te = x_te.astype(np.float16)

        result = [x_tr,x_va,x_te,A]
        data_path = 'DF_' + args.graph + str(n) + '_exp' + str(exp_id) +'.pickle'
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)


