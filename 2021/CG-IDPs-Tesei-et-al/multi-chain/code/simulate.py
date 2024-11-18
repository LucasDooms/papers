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


def simulate(residues, name, prot, temp, walltime: int | None = None):
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
    if N > 400:
        L = 25.
        Lz = 300.
        Nsteps = 2e7
    elif N > 200:
        L = 17.
        Lz = 300.
        Nsteps = 6e7
    else:
        Lz = 10 * L
        Nsteps = 6e7

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

        print(xyz[:, 0].min(), xyz[:, 0].max())
        print(xyz[:, 1].min(), xyz[:, 1].max())

        return xyz

    snapshot = hoomd.Snapshot()
    # check rank to support MPI runs with multiple processors
    if snapshot.communicator.rank == 0:
        xyz = get_3D_positions()

        snapshot.configuration.box = hoomd.Box(Lx=2.0 * L / np.cbrt(100), Ly=2.0 * L / np.cbrt(100), Lz=1.05 * abs(xyz[0, 2] - xyz[-1, 2]))
        snapshot.particles.types = types
        snapshot.bonds.types = ['polymer']
        snapshot.particles.N = N
        # resize array
        snapshot.bonds.N = N - 1

        snapshot.particles.position[:] = [[xyz[i, 0], xyz[i, 1], xyz[i, 2]] for i in range(N)]
        snapshot.particles.typeid[:] = [types.index(a) for a in fasta]
        snapshot.particles.mass[:] = [residues.loc[a].MW for a in prot.fasta]
        snapshot.particles.mass[0] += 2
        snapshot.particles.mass[-1] += 16

        snapshot.bonds.group[:] = [[i, i + 1] for i in range(N - 1)]
        snapshot.bonds.typeid[:] = [0 for _ in range(N - 1)]

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

    # replicate the system to obtain multiple chains
    nx = 4
    ny = 4
    nz = 3
    n_chains = nx * ny * nz
    simulation.state.replicate(nx=nx, ny=ny, nz=nz)

    # resize box
    ramp_steps = int(3e5)
    resize_ramp = hoomd.variant.box.Interpolate(
        initial_box=simulation.state.box,
        final_box=hoomd.Box(Lx=L, Ly=L, Lz=L),
        variant=hoomd.variant.Ramp(0.0, 1.0, simulation.timestep, ramp_steps)
    )
    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(10),
        box=resize_ramp,
    )

    integrator_method = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
    for a, mw in zip(types, MWs):
        integrator_method.gamma[str(a)] = mw / 100

    integrator = hoomd.md.Integrator(dt=0.005, methods=[integrator_method])
    integrator.forces.append(harmonic_bond)
    integrator.forces.append(lj1)
    integrator.forces.append(lj2)
    integrator.forces.append(yukawa)

    operations = hoomd.Operations()
    operations.integrator = integrator
    operations.updaters.append(box_resize)

    simulation.operations = operations

    # Logging
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(simulation, quantities=['timestep', 'walltime'])

    timelog = hoomd.write.Table(
        trigger = hoomd.trigger.Periodic(period=int(1e4)),
        logger = logger
    )
    simulation.operations.writers.append(timelog)


    gsdfile = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(1e4)),
        filename = name + "/{:d}/{:s}.gsd".format(temp, name),
        filter=hoomd.filter.All(),
        mode='wb'
    )

    simulation.operations.writers.append(gsdfile)

    # Equilibration
    simulation.run(ramp_steps + 5000)

    if snapshot.communicator.rank == 0:
        print("----------------------")
        print("Finished equilibration")
        print("----------------------")

    # remove resizer
    simulation.operations.updaters.remove(box_resize)

    snap = simulation.state.get_snapshot()
    if snap.communicator.rank == 0:

        def unwrap(d: int):
            snap_L = snap.configuration.box[d]
            for i in range(n_chains):
                new_positions = snap.particles.position[N * i:N * (i + 1)]
                for j in range(N - 1):
                    if abs(new_positions[j + 1][d] - new_positions[j][d]) > 2.0: # TODO: get sigma max
                        # move all previous points to fix periodicity
                        new_positions[:j + 1, d] += snap_L * (-1 if new_positions[j][d] > 0 else 1)

        # unwrap z-coordinate
        unwrap(2)

        # update z-direction
        snap.configuration.box = hoomd.Box(Lx=snap.configuration.box[0], Ly=snap.configuration.box[1], Lz=Lz)

    simulation.state.set_snapshot(snap)


    gsdrestart = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(1e3), phase=int(0)),
        filename=name + "/{:d}/restart.gsd".format(temp),
        filter=hoomd.filter.All(),
        mode='wb',
        truncate=True,
    )

    simulation.operations.writers.append(gsdrestart)

    # Run
    simulation.run(1e6)

    if simulation.device.communicator.rank == 0:
        print("--------------")
        print("Run completed!")
        print("--------------")

    # Make sure writers are finished
    for writer in simulation.operations.writers:
        if hasattr(writer, 'flush'):
            writer.flush()

    if simulation.device.communicator.rank == 0:
        genDCD(residues, name, prot, temp, n_chains)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', nargs='?', const='', type=str)
    parser.add_argument('--temp', nargs='?', const='', type=int)
    parser.add_argument('--walltime', nargs='?', const='', type=int, default=0)
    args = parser.parse_args()
    if args.walltime == 0:
        args.walltime = None

    print(hoomd.__file__)

    residues = pd.read_csv('residues.csv').set_index('three', drop=False)
    proteins = initProteins()
    print("protein: ", args.name, "temperature: ", args.temp)

    t0 = time.time()
    simulate(residues, args.name, proteins.loc[args.name], args.temp, args.walltime)
    print('Timing {:.3f}'.format(time.time() - t0))
