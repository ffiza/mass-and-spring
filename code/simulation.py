import itertools
import numpy as np
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

CONFIG = yaml.safe_load(open("configs/global.yml"))


def _calculate_forces(masses: np.ndarray,
                      xposs: np.ndarray,
                      yposs: np.ndarray,
                      xvels: np.ndarray,
                      yvels: np.ndarray,
                      gravitational_acceleration: float,
                      friction_coeff: float,
                      elastic_constants: np.ndarray,
                      natural_lengths: np.ndarray,
                      ) -> np.ndarray:
    """
    Calculate a 2D array with the forces of a given particle. Element
    `[i, 0]` of the array is the force acting on particle `i` in the x-axis.
    Element `[i, 1]` of the array is the force acting on particle `i` 
    in the y-axis.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.
    xvels : np.ndarray
        The x-velocities of the particles.
    yvels : np.ndarray
        The y-velocities of the particles.
    gravitational_acceleration : float
        The gravitational acceleration in the y direction in m/s^2.
    friction_coeff: float
        The friction coefficient in N*s/m.
    elastic_constants : np.ndarray
        The elastic constant of the springs in N/m.
    natural_lengths : np.ndarray
        The natural length of the springs in m.

    Returns
    -------
    forces : np.ndarray
        A 2D array with the forces of the particles.
    """
    n_bodies = len(masses)

    # Calculate the difference of position vectors
    dr = np.zeros((n_bodies, n_bodies, 2), dtype=np.float64)
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        dr[i, j] = np.array([xposs[i] - xposs[j], yposs[i] - yposs[j]])

    # Calculate the elastic forces
    forces3 = np.zeros((n_bodies, n_bodies, 2), dtype=np.float64)
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        if j > i:
            forces3[i, j] = - elastic_constants[i, j] * dr[i, j] \
                * (np.linalg.norm(dr[i, j]) - natural_lengths[i, j]) \
                / np.linalg.norm(dr[i, j])
            forces3[j, i] = - forces3[i, j]
    forces = np.sum(forces3, axis=1)

    # Add gravity
    forces[:, 1] -= np.array(masses) * gravitational_acceleration

    # Add friction
    for i in range(n_bodies):
        forces[i, :] -= friction_coeff * np.array([xvels[i], yvels[i]])

    return forces


def _calculate_potential(masses: np.ndarray,
                         xposs: np.ndarray,
                         yposs: np.ndarray,
                         gravitational_acceleration: float,
                         elastic_constants: np.ndarray,
                         natural_lengths: np.ndarray,
                         ) -> float:
    """
    Calculate a the total potential of the system at the given state.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.
    gravitational_acceleration : float
        The gravitational acceleration in the y direction in m/s^2.
    elastic_constants : np.ndarray
        The elastic constant of the springs in N/m.
    natural_lengths : np.ndarray
        The natural length of the springs in m.

    Returns
    -------
    potential : float
        The total gravitational potential energy.
    """
    n_bodies = len(masses)

    potential = 0.0
    for i in range(n_bodies):
        potential += masses[i] * gravitational_acceleration * yposs[i]
        for j in range(i + 1, n_bodies):
            dr = np.array([xposs[i] - xposs[j], yposs[i] - yposs[j]])
            potential += 0.5 * elastic_constants[i, j] \
                * (np.linalg.norm(dr) - natural_lengths[i, j])**2
    return potential


def _calculate_kinetic_energy(masses: np.ndarray,
                              xvels: np.ndarray,
                              yvels: np.ndarray,
                              ) -> float:
    """
    Calculate a the total kinetic energy of the system.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xvels : np.ndarray
        The x-velocities of the particles.
    yvels : np.ndarray
        The y-velocities of the particles.

    Returns
    -------
    float
        The total kinetic energy.
    """
    return 0.5 * np.sum(masses * (xvels**2 + yvels**2))


def simulate(masses: list,
             initial_xposs: list,
             initial_yposs: list,
             initial_xvels: list,
             initial_yvels: list,
             particle_types: list,
             elastic_constants: np.ndarray,
             natural_lengths: np.ndarray,
             gravitational_acceleration: float,
             friction_coeff: float,
             timestep: float,
             n_steps: int,
             filename: str,
             ) -> pd.DataFrame:
    """
    Perform an N-body simulation.

    Parameters
    ----------
    masses : list
        A list with the masses of the particles to be simulated.
    initial_xposs : list
        A list with the initial positions of the particles in the x-axis.
    initial_yposs : list
        A list with the initial positions of the particles in the y-axis.
    initial_xvels : list
        A list with the initial velocities of the particles in the x-axis.
    initial_yvels : list
        A list with the initial velocities of the particles in the y-axis.
    particle_types : list
        A list with `0` for the static particles and `1` for the dynamic ones.
    elastic_constants : np.ndarray
        The elastic constants of the springs in N/m.
    natural_lengths : np.ndarray
        The natural lengths of the springs in m.
    gravitational_acceleration : float
        The gravitational acceleration in the y direction in m/s^2.
    friction_coeff: float
        The friction coefficient in N*s/m.
    timestep : float
        The timestep of the simulation.
    n_steps : int
        The number of steps to simulate.
    filename : str
        The name of the simulation file.

    Returns
    -------
    pd.DataFrame
        A Pandas data frame with the results of the simulation.
    """
    # Check if all inputs have the same size
    if len(masses) != len(initial_xposs) != len(initial_yposs) != \
            len(initial_xvels) != len(initial_yvels):
        raise ValueError("All inputs must have the same size.")

    n_bodies = len(masses)

    # Make any velocity of a static particle
    for i in range(n_bodies):
        if particle_types[i] == 0:
            initial_xvels[i] = 0.0
            initial_xvels[i] = 0.0

    time = np.zeros(n_steps)

    xposs = np.zeros((n_steps, n_bodies))
    yposs = np.zeros((n_steps, n_bodies))
    xvels = np.zeros((n_steps, n_bodies))
    yvels = np.zeros((n_steps, n_bodies))

    for j in range(n_bodies):
        xposs[0, j] = initial_xposs[j]
        yposs[0, j] = initial_yposs[j]
        xvels[0, j] = initial_xvels[j]
        yvels[0, j] = initial_yvels[j]
        if particle_types[j] == 0:  # Fix static particles to initial pos
            xposs[:, j] = initial_xposs[j]
            yposs[:, j] = initial_yposs[j]

    # Integrate using the leapfrog method
    for step in tqdm(range(1, n_steps),
                     desc="Integrating...",
                     colour="#A241B2",
                     ncols=100):

        # Calculate the forces acting on the particles on previous step
        forces_then = _calculate_forces(
            masses=masses,
            xposs=xposs[step - 1],
            yposs=yposs[step - 1],
            xvels=xvels[step - 1],
            yvels=yvels[step - 1],
            friction_coeff=friction_coeff,
            gravitational_acceleration=gravitational_acceleration,
            elastic_constants=elastic_constants,
            natural_lengths=natural_lengths)
        # Update the positions
        for k in range(n_bodies):
            if particle_types[k] == 1:
                acc_then = forces_then[k] / masses[k]
                xposs[step, k] = xposs[step - 1, k] \
                    + xvels[step - 1, k] * timestep \
                    + 0.5 * acc_then[0] * timestep**2
                yposs[step, k] = yposs[step - 1, k] \
                    + yvels[step - 1, k] * timestep \
                    + 0.5 * acc_then[1] * timestep**2

        # Recalculate the force in the current step
        forces_now = _calculate_forces(
            masses=masses,
            xposs=xposs[step],
            yposs=yposs[step],
            xvels=xvels[step],
            yvels=yvels[step],
            friction_coeff=friction_coeff,
            gravitational_acceleration=gravitational_acceleration,
            elastic_constants=elastic_constants,
            natural_lengths=natural_lengths)

        # Update the velocities
        for k in range(n_bodies):
            if particle_types[k] == 1:
                acc_then = forces_then[k] / masses[k]
                acc_now = forces_now[k] / masses[k]
                xvels[step, k] = xvels[step - 1, k] \
                    + 0.5 * (acc_then[0] + acc_now[0]) * timestep
                yvels[step, k] = yvels[step - 1, k] \
                    + 0.5 * (acc_then[1] + acc_now[1]) * timestep

        # Update the time
        time[step] = time[step - 1] + timestep

    df = pd.DataFrame()
    df['Time'] = time
    for k in range(n_bodies):
        df[f"xPosition{k}"] = xposs[:, k]
        df[f"yPosition{k}"] = yposs[:, k]
        df[f"xVelocity{k}"] = xvels[:, k]
        df[f"yVelocity{k}"] = yvels[:, k]

    # Calculate energies
    kinetic_energies = np.zeros(len(df))
    potentials = np.zeros(len(df))
    for i in range(len(df)):
        kinetic_energies[i] = _calculate_kinetic_energy(
            masses=masses,
            xvels=xvels[i],
            yvels=yvels[i])
        potentials[i] = _calculate_potential(
            masses=masses,
            xposs=xposs[i],
            yposs=yposs[i],
            gravitational_acceleration=gravitational_acceleration,
            elastic_constants=elastic_constants,
            natural_lengths=natural_lengths)
    df["KineticEnergy"] = kinetic_energies
    df["Potential"] = potentials
    df["Energy"] = df["KineticEnergy"] + df["Potential"]

    df = df.round(decimals=CONFIG["DATAFRAME_DECIMALS"])

    # Sample data frame to a given FPS
    n_rows = int(CONFIG["FPS"] * df["Time"].to_numpy()[-1])
    idx = (np.linspace(
        0, 1, n_rows, endpoint=False) * len(df)).astype(np.int64)
    df = df.iloc[idx]
    df.reset_index(inplace=True, drop=True)

    df.to_csv(f"results/{filename}.csv")

    return df


def main():
    # Get IC file name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="The name of the folder that contains the configuration files.")
    args = parser.parse_args()

    # Read configuration file
    PHYSICS = yaml.safe_load(open(f"configs/{args.config}/physics.yml"))
    IC = pd.read_csv(f"configs/{args.config}/ic.csv")
    ELASTIC_CONSTANTS = np.loadtxt(
        f"configs/{args.config}/elastic_constants.csv", delimiter=',')
    NATURAL_LENGTHS = np.loadtxt(
        f"configs/{args.config}/natural_lengths.csv", delimiter=',')

    # Run simulation
    simulate(masses=IC["Mass_kg"].to_list(),
             initial_xposs=IC["xPosition_m"].to_list(),
             initial_yposs=IC["yPosition_m"].to_list(),
             initial_xvels=IC["xVelocity_m/s"].to_list(),
             initial_yvels=IC["yVelocity_m/s"].to_list(),
             particle_types=IC["DynamicParticle"].to_list(),
             elastic_constants=ELASTIC_CONSTANTS,
             natural_lengths=NATURAL_LENGTHS,
             gravitational_acceleration=PHYSICS["GRAV_ACCELERATION"],
             friction_coeff=PHYSICS["FRICTION_COEF"],
             timestep=PHYSICS["TIMESTEP"],
             n_steps=PHYSICS["N_STEPS"],
             filename=args.config)


if __name__ == "__main__":
    main()
