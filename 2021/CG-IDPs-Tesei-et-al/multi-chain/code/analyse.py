import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations


def initProteins():
    return pd.read_csv("proteins.csv", index_col=0)


def genParamsLJ(df, name, prot):
    fasta = list(prot.fasta)
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X', 'MW'] += 2
    r.loc['Z', 'MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a, 'MW'] for a in types]
    sigmamap = pd.DataFrame((r.sigmas.values + r.sigmas.values.reshape(-1, 1)
                             ) / 2, index=r.sigmas.index, columns=r.sigmas.index)
    lambdamap = pd.DataFrame((r.lambdas.values + r.lambdas.values.reshape(-1, 1)
                              ) / 2, index=r.lambdas.index, columns=r.lambdas.index)
    lj_eps = prot.eps_factor * 4.184
    # Generate pairs of amino acid types
    pairs = np.array(list(itertools.combinations_with_replacement(types, 2)))
    return pairs, lj_eps, lambdamap, sigmamap, fasta, types, MWs

def genParamsDH(df, name, prot, temp):
    kT = 8.3145 * temp * 1e-3
    fasta = list(prot.fasta)
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H', 'q'] = 1. / (1 + 10**(prot.pH - 6))
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X', 'q'] = r.loc[prot.fasta[0], 'q'] + 1.
    r.loc['Z', 'q'] = r.loc[prot.fasta[-1], 'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    qq = pd.DataFrame(r.q.values * r.q.values.reshape(-1, 1),
                      index=r.q.index, columns=r.q.index)
    def fepsw(T): return 5321 / T + 233.76 - 0.9297 * T + \
        0.1417 * 1e-2 * T * T - 0.8292 * 1e-6 * T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / kT
    charges = [r.loc[a].q * np.sqrt(lB * kT) for a in fasta]
    yukawa_eps = qq * lB * kT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8 * np.pi * lB * prot.ionic * 6.022 / 10)
    return yukawa_eps, yukawa_kappa, charges


def genDCD(residues, name, prot, temp, n_chains):
    top = md.Topology()
    for _ in range(n_chains):
        chain = top.add_chain()
        for resname in prot.fasta:
            residue = top.add_residue(residues.loc[resname, 'three'], chain)
            top.add_atom('CA', element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms - 1):
            top.add_bond(chain.atom(i), chain.atom(i + 1))

    print("READING FILE")
    print(name + '/{:d}'.format(temp) + '/{:s}.gsd'.format(name))

    t = md.load(name + '/{:d}'.format(temp) + '/{:s}.gsd'.format(name))
    t.top = top

    # convert particle positions and box side lengths to Å to simplify operations to center the slab in the box
    # the bin width of the number density profiles is 1 Å
    t.xyz *= 10
    t.unitcell_lengths *= 10

    lz = t.unitcell_lengths[0, 2]
    edges = np.arange(-lz / 2., lz / 2., 1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz

    # calculate per-frame number density profiles
    h = np.apply_along_axis(lambda a: np.histogram(a, bins=edges)[0], 1, t.xyz[:, :, 2])

    # find midpoint of protein-rich region
    zmid = np.apply_along_axis(lambda a: z[a > np.quantile(a, .98)].mean(), 1, h)
    # find atom closest to midpoint in each frame
    indices = np.argmin(np.abs(t.xyz[:, :, 2] - zmid[:, np.newaxis]), axis=1)

    t[0].save_gsd(name + "/{:d}".format(temp) + '/topology.gsd')
    # t[0].save_pdb(name + "/{:d}".format(temp) + '/top.pdb')
    t.save_dcd(name + "/{:d}".format(temp) + '/traj2.dcd')

    # translate each frame so that atom closest to midpoint is at z = 0
    u = MDAnalysis.Universe(
        name + "/{:d}".format(temp) + '/topology.gsd',
        name + '/{:d}'.format(temp) + '/traj2.dcd',
    )

    ag = u.atoms
    with MDAnalysis.Writer(name + '/{:d}'.format(temp) + '/traj1.dcd', u.atoms.n_atoms) as W:
        for ts, ndx in zip(u.trajectory, indices):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms(
                    'index {:d}'.format(ndx)),
                center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)

            W.write(u.atoms)

    t = md.load(
        name + '/{:d}'.format(temp) + '/traj1.dcd',
        top=name + '/{:d}'.format(temp) + '/topology.gsd'
    )
    lz = t.unitcell_lengths[0, 2]
    edges = np.arange(0, lz, 1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    h = np.apply_along_axis(lambda a: np.histogram(
        a, bins=edges)[0], 1, t.xyz[:, :, 2])
    h1 = np.mean(h[500:], axis=0)

    # find displacement along z that maximizes correlation function
    # between instantaneous and time-averaged profiles
    maxoverlap = np.apply_along_axis(
        lambda a: np.correlate(h1, np.histogram(a, bins=edges)[0], 'full').argmax() - h1.size + dz,
        1, t.xyz[:, :, 2]
    )

    # translate by displacement along z that maximizes correlation function
    u = MDAnalysis.Universe(
        name + '/{:d}'.format(temp) + '/topology.gsd',
        name + '/{:d}'.format(temp) + '/traj1.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(name + '/{:d}'.format(temp) + '/traj.dcd', ag.n_atoms) as W:
        for ts, mo in zip(u.trajectory, maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0, 0, mo * 10])(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(
        name + '/{:d}'.format(temp) + '/traj.dcd',
        top=name + '/{:d}'.format(temp) + '/topology.gsd'
    )
    h = np.apply_along_axis(lambda a: np.histogram(
        a, bins=edges)[0], 1, t.xyz[:, :, 2])
    np.save('{:s}_{:d}.npy'.format(name, temp), h, allow_pickle=False)
    os.remove(name + '/{:d}'.format(temp) + '/traj1.dcd')
    os.remove(name + '/{:d}'.format(temp) + '/traj2.dcd')

    # convert to nanometers and save
    t.xyz /= 10
    t.unitcell_lengths /= 10
    t[0].save_pdb(name + '/{:d}'.format(temp) + '/topology.gsd')
    # t[0].save_pdb(name + '/{:d}'.format(temp) + '/top.pdb')
    t.save_dcd(name + '/{:d}'.format(temp) + '/traj.dcd')
