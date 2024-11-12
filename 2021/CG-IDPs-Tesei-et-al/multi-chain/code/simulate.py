from analyse import initProteins, genDCD, genParamsLJ, genParamsDH
import hoomd
import hoomd.md
import time
from argparse import ArgumentParser
from mdtraj.utils.rotation import rotation_matrix_from_quaternion
from PeptideBuilder import Geometry
import PeptideBuilder
from Bio.PDB.PDBIO import PDBIO
import numpy as np
import pandas as pd


def simulate(residues, name, prot, temp):
    residues = residues.set_index('one')

    try:
        device = hoomd.device.GPU()
    except Exception as e:
        print("GPU initialisation returned an error:")
        print(e)
        print("")
        print("Attempting CPU initialisation")
        print("")
        device = hoomd.device.CPU()
    else:
        if not device.is_available():
            print("GPU not available, running on CPU instead!")
            device = hoomd.device.CPU()

    device.notice_level = 1

    simulation = hoomd.Simulation(device, seed=40495)

    pairs, lj_eps, lj_lambda, lj_sigma, fasta, types, MWs = genParamsLJ(residues, name, prot)
    yukawa_eps, yukawa_kappa, _ = genParamsDH(residues, name, prot, temp)

    # Protein length (number of amino acids)
    N = len(fasta)

    L = 15.
    margin = 2
    if N > 400:
        L = 25.
        Lz = 300.
        margin = 8
        Nsteps = 2e7
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
        Nsteps = 6e7
    else:
        Lz = 10 * L
        Nsteps = 6e7

    def get_xy_positions(n_chains_max: int = 100):
        """Generate random position in a 2D box"""
        xy = np.empty(0)
        xy = np.append(xy, np.random.rand(2) * (L - margin) - (L - margin) / 2).reshape((-1, 2))

        for x, y in np.random.rand(1000, 2) * (L - margin) - (L - margin) / 2:
            # check if too close to existing positions
            if np.any(np.linalg.norm(xy - [x, y], axis=1) <= .7):
                continue

            # same for periodic images
            x_periodic_image = x - L if x > 0 else x + L
            if np.any(np.linalg.norm(xy - [x_periodic_image, y], axis=1) <= .7):
                continue
            y_periodic_image = y - L if y > 0 else y + L
            if np.any(np.linalg.norm(xy - [x, y_periodic_image], axis=1) <= .7):
                continue
            if np.any(np.linalg.norm(xy - [x_periodic_image, y_periodic_image], axis=1) <= .7):
                continue

            xy = np.append(xy, [x, y]).reshape((-1, 2))

            if xy.shape[0] == n_chains_max:
                break

        return xy

    xy = get_xy_positions()
    n_chains = xy.shape[0]

    print(f'Number of chains {n_chains}, {N} residues long')

    def get_3D_positions():
        geo = Geometry.geometry(prot.fasta[0])
        geo.phi = -120
        geo.psi_im1 = 150
        structure = PeptideBuilder.initialize_res(geo)
        for residue in prot.fasta[1:]:
            structure = PeptideBuilder.add_residue(
                structure, residue, geo.phi, geo.psi_im1
            )

        out = PDBIO()
        out.set_structure(structure)
        xyz = []
        for atom in out.structure.get_atoms():
            if atom.name == 'CA':
                xyz.append(atom.coord[:3])
        xyz = np.array(xyz) / 10.0

        v = xyz[-1] - xyz[0]
        u = np.array([0, 0, 1])
        a = np.cross(v, u)
        a = a / np.linalg.norm(a, keepdims=True)
        b = np.arccos(np.dot(v, u) / np.linalg.norm(v))
        quaternion = np.insert(np.sin(-b / 2).reshape(-1, 1) * a, 0, np.cos(-b / 2), axis=1)
        newxyz = xyz - np.mean(xyz, axis=0)
        newxyz = np.matmul(newxyz, rotation_matrix_from_quaternion(quaternion))
        xyz = np.array(newxyz[0])

        print(xyz[:, 0].min(), xyz[:, 0].max(), xy[:, 0].min(), xy[:, 0].max())
        print(xyz[:, 1].min(), xyz[:, 1].max(), xy[:, 1].min(), xy[:, 1].max())
        
        return xyz

    snapshot = hoomd.Snapshot()
    # check rank to support MPI runs with multiple processors
    if snapshot.communicator.rank == 0:
        snapshot.configuration.box = hoomd.Box(Lx=L, Ly=L, Lz=Lz)
        snapshot.particles.types = types
        snapshot.bonds.types = ['polymer']
        snapshot.particles.N = N * n_chains
        # resize array
        snapshot.bonds.N = n_chains * (N - 1)

        xyz = get_3D_positions()

        for j, (x, y) in enumerate(xy):
            begin = j * N
            end = j * N + N

            snapshot.particles.position[begin:end] = [
                [xyz[i, 0] + x, xyz[i, 1] + y, xyz[i, 2]] for i in range(N)]
            snapshot.particles.typeid[begin:end] = [types.index(a) for a in fasta]
            snapshot.particles.mass[begin:end] = [
                residues.loc[a].MW for a in prot.fasta]
            snapshot.particles.mass[begin] += 2
            snapshot.particles.mass[end - 1] += 16

            snapshot.bonds.group[begin - j:end - j - 1] = [
                [i, i + 1] for i in range(begin, end - 1)
            ]
            snapshot.bonds.typeid[begin - j:end - j - 1] = [0] * (N - 1)

    simulation.create_state_from_snapshot(snapshot)

    kT = 8.3145 * temp * 1e-3
    harmonic_bond = hoomd.md.bond.Harmonic()
    harmonic_bond.params['polymer'] = {'k': 8033.0, 'r0': 0.38}

    neighbor_list = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond',), deterministic=True)

    lj1 = hoomd.md.pair.LJ(nlist=neighbor_list, default_r_cut=4.0, mode='shift')
    lj2 = hoomd.md.pair.LJ(nlist=neighbor_list, default_r_cut=4.0)
    yukawa = hoomd.md.pair.Yukawa(nlist=neighbor_list, default_r_cut=4.0, mode='shift')
    for a, b in pairs:
        yukawa.params[(str(a), str(b))] = {'epsilon': yukawa_eps.loc[a, b], 'kappa': yukawa_kappa}

        lj1.params[(str(a), str(b))] = {'epsilon': lj_eps * (1 - lj_lambda.loc[a, b]), 'sigma': lj_sigma.loc[a, b]}
        lj1.r_cut[(str(a), str(b))] = 2.**(1./6.) * lj_sigma.loc[a, b]

        lj2.params[(str(a), str(b))] = {'epsilon': lj_eps * lj_lambda.loc[a, b], 'sigma': lj_sigma.loc[a, b]}

    walls = []
    if N > 400:
        walls.append(
            hoomd.wall.Plane((0, 0, -50), (0, 0, 1))
        )
        walls.append(
            hoomd.wall.Plane((0, 0, 50), (0, 0, -1))
        )
    elif N > 200:
        walls.append(
            hoomd.wall.Plane((0, 0, -30), (0, 0, 1))
        )
        walls.append(
            hoomd.wall.Plane((0, 0, 30), (0, 0, -1))
        )
    else:
        walls.append(
            hoomd.wall.Plane((0, 0, -10), (0, 0, 1))
        )
        walls.append(
            hoomd.wall.Plane((0, 0, 10), (0, 0, -1))
        )

    gaussian_wall = hoomd.md.external.wall.Gaussian(walls)
    gaussian_wall.params[types] = {'epsilon': 10.0, 'sigma': 1.0, 'r_cut': 4.0}

    integrator_method = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
    for a, mw in zip(types, MWs):
        integrator_method.gamma[str(a)] = mw / 100

    integrator = hoomd.md.Integrator(dt=0.005, methods=[integrator_method])
    integrator.forces.append(harmonic_bond)
    integrator.forces.append(lj1)
    integrator.forces.append(lj2)
    integrator.forces.append(yukawa)
    integrator.forces.append(gaussian_wall)

    operations = hoomd.Operations()
    operations.integrator = integrator

    simulation.operations = operations

    # Logging
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(simulation, quantities=['timestep', 'walltime'])

    timelog = hoomd.write.Table(
        trigger = hoomd.trigger.Periodic(period=int(1e3)),
        logger = logger
    )
    simulation.operations.writers.append(timelog)

    # Equilibration
    simulation.run(2e7)

    if snapshot.communicator.rank == 0:
        print("----------------------")
        print("Finished equilibration")
        print("----------------------")

    # Remove walls
    simulation.operations.integrator.forces.remove(gaussian_wall)

    gsdfile = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(5e4)),
        filename = name + "/{:d}/{:s}.gsd".format(temp, name),
        filter=hoomd.filter.All(),
        mode='wb'
    )
    gsdrestart = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(1e6), phase=int(0)),
        filename=name + "/{:d}/restart.gsd".format(temp),
        filter=hoomd.filter.All(),
        mode='wb',
        truncate=True,
    )

    simulation.operations.writers.append(gsdfile)
    simulation.operations.writers.append(gsdrestart)

    # Run
    simulation.run(Nsteps)

    if snapshot.communicator.rank == 0:
        print("--------------")
        print("Run completed!")
        print("--------------")

    # Make sure writers are finished
    for writer in simulation.operations.writers:
        if hasattr(writer, 'flush'):
            writer.flush()

    if snapshot.communicator.rank == 0:
        genDCD(residues, name, prot, temp, n_chains)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', nargs='?', const='', type=str)
    parser.add_argument('--temp', nargs='?', const='', type=int)
    args = parser.parse_args()

    print(hoomd.__file__)

    residues = pd.read_csv('residues.csv').set_index('three', drop=False)
    proteins = initProteins()
    print("protein: ", args.name, "temperature: ", args.temp)

    t0 = time.time()
    simulate(residues, args.name, proteins.loc[args.name], args.temp)
    print('Timing {:.3f}'.format(time.time() - t0))
