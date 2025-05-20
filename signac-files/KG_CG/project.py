"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
import pickle
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
from unyt import Unit


class KGCG(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpu-v100",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="v100," "batch",
            help="Specify the partition to submit to."
        )


@KGCG.label
def system_built(job):
    return job.isfile("init_frame.gsd")


@KGCG.label
def initial_run_done(job):
    return job.doc.runs > 0


@KGCG.label
def equilibrated(job):
    return job.doc.equilibrated


@KGCG.label
def sampled(job):
    return job.doc.sampled


@KGCG.label
def production_done(job):
    return job.isfile("production-restart.gsd")

@KGCG.post(system_built)
@KGCG.operation(
    directives={"ngpu": 1, "ncpu": 1, "executable": "python -u"}, name="build"
)
def build(job):
    """Build system."""
    import mbuild as mb
    from mbuild.path import HardSphereRandomWalk
    from mbuild import Polymer
    import numpy as np
    from scipy.spatial.distance import pdist
    import freud
    
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

    chain_length = job.doc.lengths
    n_chains = job.doc.num_mols
    
    seeds = [np.random.randint(low=1) for i in range(n_chains)]
    chains = []
    
    for seed in seeds:
        rw_path = HardSphereRandomWalk(
            N=chain_length,
            bond_length=1.12,
            radius=1.1,
            min_angle=np.pi/2,
            max_angle=2.5,
            max_attempts=1e4,
            seed=seed,
        )
        rw_path.generate()
        chains.append(rw_path.to_compound(bead_name="A", bead_mass=1.0))
    a = int(chain_length//2 + 25)
    box = mb.fill_box(compound=chains, n_compounds=[1 for i in chains], box=[a,a,a],overlap=2,edge=3)
    box.check_for_overlap(minimum_distance=1.0, excluded_bond_depth=1)
    box.save(job.fn("init_frame.gsd"))

@KGCG.pre(system_built)
@KGCG.post(initial_run_done)
@KGCG.operation(
    directives={"ngpu": 1, "ncpu": 1, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial simulation."""
    import unyt
    from unyt import Unit
    import numpy
    import flowermd
    from flowermd.base import Simulation
    from flowermd.library import KremerGrestBeadSpring
    from flowermd.utils import get_target_box_number_density
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")


        ff = KremerGrestBeadSpring(bond_k=100,bond_max=1.15,sigma=1.0)
        gsd_path = job.fn(f"trajectory{job.doc.runs}.gsd")
        log_path = job.fn(f"log{job.doc.runs}.txt")
        seed = numpy.random.randint(1,1e4)
        sim = Simulation(
            initial_state=job.fn("init_frame.gsd"),
            forcefield=ff.hoomd_forces,
            dt=job.sp.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=seed,
        )

        target_box = get_target_box_number_density(density=job.sp.density*Unit("nm**-3"),n_beads=job.doc.num_mols*job.doc.lengths)
        sim.pickle_forcefield(job.fn("forcefield.pickle"))
        # Store more unit information in job doc
        tau_kT = job.sp.dt * 100
        job.doc.tau_kT = tau_kT
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"
        job.doc.target_box = target_box.value
        job.doc.seed = seed
    
        sim.run_update_volume(
            final_box_lengths=target_box,
            kT=3.0,
            n_steps=1e5,
            tau_kt=100*job.sp.dt,
            period=10,
            thermalize_particles=True
        )
        sim.save_restart_gsd(job.fn("shrink_restart.gsd"))
        print("Shrinking simulation finished...")
        sim.run_NVT(n_steps=job.sp.n_equil_steps, kT=job.sp.kT, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs = 1
        print("Simulation finished.")

@KGCG.pre(initial_run_done)
@KGCG.post(equilibrated)
@KGCG.operation(
    directives={"ngpu": 1, "ncpu": 1, "executable": "python -u"},
    name="run-longer"
)
def run_longer(job):
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            hoomd_ff = pickle.load(f)
        gsd_path = job.fn(f"trajectory{job.doc.runs}.gsd")
        log_path = job.fn(f"log{job.doc.runs}.txt")
        sim = Simulation(
            initial_state=job.fn("restart.gsd"),
            forcefield=hoomd_ff,
            dt=job.sp.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.doc.seed,
        )
        print("Running simulation.")
        sim.run_NVT(
            n_steps=job.sp.n_equil_steps,
            kT=job.sp.kT,
            tau_kt=job.doc.tau_kT,
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs += 1
        print("Simulation finished.")

@KGCG.pre(equilibrated)
@KGCG.post(production_done)
@KGCG.operation(
    directives={"ngpu": 1, "ncpu": 1, "executable": "python -u"},
    name="production"
)
def production_run(job):
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        print("Running the production run...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            hoomd_ff = pickle.load(f)

        gsd_path = job.fn(f"production.gsd")
        log_path = job.fn(f"production.txt")

        sim = Simulation(
            initial_state = job.fn("restart.gsd"),
            forcefield=hoomd_ff,
            dt=job.sp.dt,
            gsd_write_freq=int(5e5),
        gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.doc.seed,
        )
        print("Running simulation.")
        sim.run_NVT(
            n_steps=job.sp.n_prod_steps*2,
            kT=job.sp.kT,
            tau_kt=job.doc.tau_kT,
        )
        sim.save_restart_gsd(job.fn("production-restart.gsd"))
        job.doc.production_runs += 1
        print("Simulation finished.")
   

@KGCG.pre(production_done)
@KGCG.operation(
    directives={"ngpu": 1, "ncpu": 1, "executable": "python -u"},
    name="production_run_longer"
)
def production_run_longer(job):
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        print("Continuing the production run...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            hoomd_ff = pickle.load(f)

        gsd_path = job.fn(f"production{job.doc.production_runs+1}.gsd")
        log_path = job.fn(f"production{job.doc.production_runs+1}.txt")

        sim = Simulation(
            initial_state=job.fn("production-restart.gsd"),
            forcefield=hoomd_ff,
            dt=job.sp.dt,
            gsd_write_freq=int(5e5),
        gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.doc.seed,
        )
        print("Running simulation.")
        sim.run_NVT(
            n_steps=job.sp.n_prod_steps*2,
            kT=job.sp.kT,
            tau_kt=job.doc.tau_kT,
        )
        sim.save_restart_gsd(job.fn("production-restart.gsd"))
        print("Simulation finished.")
        job.doc.production_runs += 1

@KGCG.pre(production_done)
@KGCG.operation(
    directives={"ngpu": 0, "ncpu": 1, "executable": "python -u"},
    name="sample"
)
def sample(job):
    import numpy as np
    from cmeutils.dynamics import msd_from_gsd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        steps_per_frame = int(5e5)
        # Update job doc
        ts = job.doc.real_time_step * 1e-15
        ts_frame = steps_per_frame * ts
    
        msd = msd_from_gsd(
                gsdfile=job.fn("production-combined-center.gsd"),
                start=0,
                stop=-1,
                atom_types="B",
                msd_mode="direct"
        )
        msd_results = np.copy(msd.msd)
        time_array = np.arange(0, len(msd.msd), 1) * ts_frame
        np.save(file=job.fn(f"msd_time_mid_c.npy"), arr=time_array)
        np.save(file=job.fn(f"msd_data_reduced_mid_c.npy"), arr=msd.msd)
        
        print("Finished.")
        job.doc.sampled = True
        
if __name__ == "__main__":
    KGCG(environment=Fry).main()
