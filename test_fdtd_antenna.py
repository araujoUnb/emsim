from antennas.rectangular_patch import RectangularPatch
from simulator.fdtd_antenna import FDTDAntenna
from simulator.postprocessor import PostProcessor
import os

def main():
    # Simulation path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(project_root, "data", "test")

    # Create the antenna
    patch = RectangularPatch(feed_pos=-6)

    # Create FDTD simulation object
    sim = FDTDAntenna(patch, sim_path=sim_path, f0=2e9, fc=1e9)

    # Build the simulation
    sim.configure_simulation()
    sim.setup_simulation_domain()
    sim.add_geometry_to_grid()
    sim.add_feed_port(feed_R=50)
    sim.add_nf2ff_box()
    sim.export_to_xml("simp_patch.xml")
    sim.run_simulation()

    # Post-processing
    post = PostProcessor(sim_path, sim.port, sim.get_nf2ff_box(), patch, f_start=1e9, f_stop=3e9)
    post.plot_s11()
    post.plot_impedance()
    nf2ff_result, freq = post.compute_nf2ff(center=[0, 0, 1e-3])
    post.plot_radiation_pattern(nf2ff_result, freq)
    #post.plot_3d_radiation(nf2ff_result)

if __name__ == "__main__":
    main()
