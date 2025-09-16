import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5.0, loc_std=0.5, vel_norm=0.5, interaction_strength=0.1, noise_var=0.0):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0.0, 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            K = 0.5 * (vel**2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r**2).sum())
                        U += 0.5 * self.interaction_strength * edges[i, j] * (dist**2) / 2
            return U + K

    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10, spring_prob=[1.0 / 2, 0, 1.0 / 2]):
        n = self.n_balls
        assert T % sample_freq == 0
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.random.choice(self._spring_types, size=(self.n_balls, self.n_balls), p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next**2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            forces_size = -self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(1, n, n),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(1, n, n),
                    )
                )
            ).sum(axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = -self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(1, n, n),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(1, n, n),
                        )
                    )
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=str, default="springs", help="What simulation to generate.")
    parser.add_argument("--num_all", type=int, default=10000, help="Number of total simulations to generate.")
    parser.add_argument("--num-train", type=int, default=50000, help="Number of training simulations to generate.")
    parser.add_argument("--num-valid", type=int, default=10000, help="Number of validation simulations to generate.")
    parser.add_argument("--num-test", type=int, default=10000, help="Number of test simulations to generate.")
    parser.add_argument("--length", type=int, default=5000, help="Length of trajectory.")
    parser.add_argument("--length-test", type=int, default=10000, help="Length of test set trajectory.")
    parser.add_argument("--sample-freq", type=int, default=100, help="How often to sample the trajectory.")
    parser.add_argument("--n-balls", type=int, default=10, help="Number of balls in the simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length, sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all


if __name__ == "__main__":
    args = parse_args()
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = "_springs"
    suffix += str(args.n_balls)
    suffix += "_" + str(args.sample_freq)
    np.random.seed(args.seed)
    print("Generating {} data with {} balls.".format(args.simulation, args.n_balls))
    t = time.time()
    print("Generating {} total simulations".format(args.num_all))
    loc_all, vel_all, edges_all = generate_dataset(args.num_all, args.length, args.sample_freq)
    print("Total data generation time: {}".format(time.time() - t))
    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_all.shape[3]

    loc_max = loc_all.max()
    loc_min = loc_all.min()
    vel_max = vel_all.max()
    vel_min = vel_all.min()

    # Normalize to [-1, 1]
    loc_all = 2 * (loc_all - loc_min) / (loc_max - loc_min) - 1
    vel_all = 2 * (vel_all - vel_min) / (vel_max - vel_min) - 1
    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_all = loc_all.transpose(0, 3, 1, 2)
    vel_all = vel_all.transpose(0, 3, 1, 2)
    feat_all = np.concatenate([loc_all, vel_all], axis=3)
    # Reshape to [num_sims, num_timesteps, num_atoms, num_dims*2]
    feat_all = feat_all.transpose(0, 2, 1, 3)
    print("Final data shape: ", feat_all.shape)
    print("Final edges shape: ", edges_all.shape)

    # save feat.npy and edges.npy
    np.save("feat" + suffix + ".npy", feat_all)
    np.save("edges" + suffix + ".npy", edges_all)
