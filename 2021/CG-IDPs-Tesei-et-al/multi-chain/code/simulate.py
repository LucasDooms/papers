from analyse import initProteins, genParamsLJ, genParamsDH
import hoomd
import hoomd.md
import time
from mdtraj.utils.rotation import rotation_matrix_from_quaternion
from PeptideBuilder import Geometry
import PeptideBuilder
from Bio.PDB.PDBIO import PDBIO
import numpy as np
import pandas as pd
import flow


def create_initial_system(N: int, residues, prot, fasta, types, L: float, Lz: float, margin: float):
    def get_xy_positions(n_chains_max: int = 40):
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
        for atom in out.structure.get_atoms(): # type: ignore
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
        snapshot.configuration.box = hoomd.Box(Lx=L, Ly=L, Lz=Lz) # type: ignore
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

    return snapshot


class Project(flow.FlowProject):
    pass


def get_device(silent: bool = False) -> hoomd.device.GPU | hoomd.device.CPU:

    def silent_print(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    try:
        device = hoomd.device.GPU()
    except Exception as e:
        silent_print("GPU initialisation returned an error:")
        silent_print(e)
        silent_print("")
        silent_print("Attempting CPU initialisation")
        silent_print("")
        device = hoomd.device.CPU()
    else:
        if not device.is_available():
            silent_print("GPU not available, running on CPU instead!")
            device = hoomd.device.CPU()

    device.notice_level = 1
    silent_print("Obtained device: running on", device)
    return device


def create_simulation(residues, name, prot, temp, model, seed) -> hoomd.Simulation:
    """Creates a simulation with the default interactions (harmonic, Asbaugh-Hatch, Debye-Huckel).
    The state must still be loaded after calling this function (e.g. from a gsd file)."""
    device = get_device()
    simulation = hoomd.Simulation(device, seed=seed)

    pairs, lj_eps, lj_lambda, lj_sigma, fasta, types, MWs = genParamsLJ(residues, name, prot, model)
    yukawa_eps, yukawa_kappa, _ = genParamsDH(residues, name, prot, temp)

    kT = 8.3145 * temp * 1e-3
    harmonic_bond = hoomd.md.bond.Harmonic()
    harmonic_bond.params['polymer'] = {'k': 8033.0, 'r0': 0.38} # type: ignore

    neighbor_list = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond',), deterministic=True)

    lj1 = hoomd.md.pair.LJ(nlist=neighbor_list, default_r_cut=4.0, mode='shift')
    lj2 = hoomd.md.pair.LJ(nlist=neighbor_list, default_r_cut=4.0)
    yukawa = hoomd.md.pair.Yukawa(nlist=neighbor_list, default_r_cut=4.0, mode='shift')
    for a, b in pairs:
        yukawa.params[(str(a), str(b))] = {'epsilon': yukawa_eps.loc[a, b], 'kappa': yukawa_kappa} # type: ignore

        lj1.params[(str(a), str(b))] = {'epsilon': lj_eps * (1 - lj_lambda.loc[a, b]), 'sigma': lj_sigma.loc[a, b]} # type: ignore
        lj1.r_cut[(str(a), str(b))] = 2.**(1./6.) * lj_sigma.loc[a, b] # type: ignore

        lj2.params[(str(a), str(b))] = {'epsilon': lj_eps * lj_lambda.loc[a, b], 'sigma': lj_sigma.loc[a, b]} # type: ignore

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

    simulation.operations = operations

    return simulation


@Project.post.true("initialized") # type: ignore
@Project.operation # type: ignore
def initialize(job):
    residues = pd.read_csv('residues.csv').set_index('one')
    proteins = initProteins()
    name = job.sp.name
    prot = proteins.loc[name]
    temp = job.sp.temp
    model = job.sp.model

    pairs, lj_eps, lj_lambda, lj_sigma, fasta, types, MWs = genParamsLJ(residues, name, prot, model)
    yukawa_eps, yukawa_kappa, _ = genParamsDH(residues, name, prot, temp)

    # Protein length (number of amino acids)
    N = len(fasta)
    job.doc.fasta = fasta
    job.doc.types = types
    job.doc.N = N

    L = 15.
    margin = 2
    if N > 400:
        L = 25.
        Lz = 300.
        margin = 8
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
    else:
        Lz = 10 * L

    job.doc.L = L
    job.doc.Lz = Lz
    job.doc.margin = margin

    snapshot = create_initial_system(N, residues, prot, fasta, types, L, Lz, margin)

    simulation = create_simulation(residues, name, prot, temp, model, job.sp.seed)
    simulation.create_state_from_snapshot(snapshot)

    hoomd.write.GSD.write(simulation.state, job.fn("initial.gsd"))

    job.doc.initialized = True


def run_with_restart(simulation, end_step: int, job, restart_file_name: str, finish, steps_per_loop: int = int(1e5)):
    try:
        while simulation.timestep < end_step:
            print(simulation.timestep)
            simulation.run(min(steps_per_loop, end_step - simulation.timestep))

            # The walltime will be reached soon, abort now
            if simulation.walltime + simulation.device.communicator.walltime >= job.sp.walltime:
                break
        else:
            # finished entire simulation
            finish()
    finally:
        hoomd.write.GSD.write(
            state=simulation.state, mode="wb", filename=job.fn(restart_file_name)
        )
        job.document["timestep"] = simulation.timestep

        walltime = simulation.device.communicator.walltime
        simulation.device.notice(
            f"{job.id} ended on step {simulation.timestep} after {walltime} seconds"
        )


N_EQUILIBRIUM_STEPS: int = int(1e6)
EQUILIBRIUM_FN: str = "equilibrated.gsd"
EQUILIBRIUM_RESTART_FN: str = "restart_equilibrium.gsd"

@Project.pre.true("initialized") # type: ignore
@Project.post.true("equilibrated") # type: ignore
@Project.operation(directives={"ngpu": 1}) # type: ignore
def equilibrate(job):
    end_step = N_EQUILIBRIUM_STEPS

    residues = pd.read_csv('residues.csv').set_index('one')
    proteins = initProteins()
    name = job.sp.name
    prot = proteins.loc[name]
    temp = job.sp.temp

    simulation = create_simulation(residues, name, prot, temp, job.sp.model, job.sp.seed)
    simulation.create_state_from_gsd(
        job.fn(EQUILIBRIUM_RESTART_FN if job.isfile(EQUILIBRIUM_RESTART_FN) else "initial.gsd")
    )

    walls = []
    if job.doc.N > 400:
        walls.append(
            hoomd.wall.Plane((0, 0, -50), (0, 0, 1))
        )
        walls.append(
            hoomd.wall.Plane((0, 0, 50), (0, 0, -1))
        )
    elif job.doc.N > 200:
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
    gaussian_wall.params[job.doc.types] = {'epsilon': 10.0, 'sigma': 1.0, 'r_cut': 4.0} # type: ignore

    simulation.operations.integrator.forces.append(gaussian_wall) # type: ignore


    gsdrestart = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(1e5), phase=int(0)),
        filename= job.fn(EQUILIBRIUM_RESTART_FN),
        filter= hoomd.filter.All(),
        mode='wb',
        truncate=True,
    )

    simulation.operations.writers.append(gsdrestart)

    def finish():
        if simulation.device.communicator.rank == 0:
            print("----------------------")
            print("Finished equilibration")
            print("----------------------")

        # Remove walls
        simulation.operations.integrator.forces.remove(gaussian_wall) # type: ignore

        hoomd.write.GSD.write(simulation.state, job.fn(EQUILIBRIUM_FN))

        job.document.equilibrated_step = simulation.timestep
        job.document.equilibrated = True

    # Equilibration
    run_with_restart(simulation, end_step, job, EQUILIBRIUM_RESTART_FN, finish)


RESTART_FN: str = "restart.gsd"

@Project.pre.true("equilibrated") # type: ignore
@Project.post.true("finished") # type: ignore
@Project.operation(directives={"ngpu": 1}) # type: ignore
# residues, name, prot, temp, walltime: int | None = None, is_restart: bool = False
def simulate(job):
    residues = pd.read_csv('residues.csv').set_index('one')
    proteins = initProteins()
    name = job.sp.name
    prot = proteins.loc[name]
    temp = job.sp.temp
    Nsteps = job.sp.Nsteps

    end_step = job.document["equilibrated_step"] + Nsteps

    simulation = create_simulation(residues, name, prot, temp, job.sp.model, job.sp.seed)
    simulation.create_state_from_gsd(
        job.fn(RESTART_FN if job.isfile(RESTART_FN) else EQUILIBRIUM_FN)
    )

    # Logging
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(simulation, quantities=['timestep', 'walltime'])

    timelog = hoomd.write.Table(
        trigger = hoomd.trigger.Periodic(period=int(1e3)),
        logger = logger
    )
    simulation.operations.writers.append(timelog)

    gsdfile = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(5e4)),
        filename = job.fn("run.gsd"),
        filter= hoomd.filter.All(),
        mode='wb'
    )
    gsdrestart = hoomd.write.GSD(
        trigger = hoomd.trigger.Periodic(period=int(1e6), phase=int(0)),
        filename= job.fn(RESTART_FN),
        filter= hoomd.filter.All(),
        mode='wb',
        truncate=True,
    )

    simulation.operations.writers.append(gsdfile)
    simulation.operations.writers.append(gsdrestart)

    def finish():
        if simulation.device.communicator.rank == 0:
            print("--------------")
            print("Run completed!")
            print("--------------")

        # Make sure writers are finished
        for writer in simulation.operations.writers:
            if hasattr(writer, 'flush'):
                writer.flush()

        job.document.finished = True

    # Run
    run_with_restart(simulation, end_step, job, RESTART_FN, finish)


# The simulation can be started with `python simulate.py run` or submitted to a cluster with `python simulate.py submit`
def main():
    project = Project()
    t0 = time.time()
    project.main()
    print('Timing {:.3f}'.format(time.time() - t0))
    project.print_status(
        overview=False, detailed=True, hide_progress=True, parameters=["temp"]
    )


if __name__ == "__main__":
    main()
