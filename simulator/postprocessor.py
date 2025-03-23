import numpy as np
import plotly.graph_objects as go

class PostProcessor:
    def __init__(self, sim_path, port, nf2ff_box=None, antenna=None, f_start=1e9, f_stop=3e9, npoints=401):
        self.sim_path = sim_path
        self.port = port
        self.nf2ff_box = nf2ff_box
        self.antenna = antenna
        self.f = np.linspace(f_start, f_stop, npoints)
        self.port.CalcPort(sim_path, self.f)

    def plot_s11(self):
        s11 = self.port.uf_ref / self.port.uf_inc
        s11_dB = 20 * np.log10(np.abs(s11))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.f / 1e9, y=s11_dB, mode='lines', name='S11 (dB)'))
        fig.update_layout(title="S11 (Reflection Coefficient)",
                          xaxis_title="Frequency (GHz)",
                          yaxis_title="|S11| (dB)",
                          template="plotly_white")
        fig.show()

    def plot_impedance(self):
        Zin = self.port.uf_tot / self.port.if_tot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.f / 1e9, y=np.real(Zin), mode='lines', name='Re{Zin}'))
        fig.add_trace(go.Scatter(x=self.f / 1e9, y=np.imag(Zin), mode='lines', name='Im{Zin}', line=dict(dash='dash')))
        fig.update_layout(title="Input Impedance",
                          xaxis_title="Frequency (GHz)",
                          yaxis_title="Impedance (Ohm)",
                          template="plotly_white")
        fig.show()

    def compute_nf2ff(self, theta=np.arange(-180, 181, 2), phi=[0, 90], center=[0, 0, 1e-3]):
        s11 = self.port.uf_ref / self.port.uf_inc
        s11_dB = 20 * np.log10(np.abs(s11))
        idx = np.argmin(np.abs(s11_dB))
        freq = self.f[idx]
        
        print(f"📡 Calculating NF2FF at {freq/1e9:.2f} GHz")

        outfile = "nf2ff_result.h5"
        nf2ff_result = self.nf2ff_box.CalcNF2FF(
            self.sim_path, outfile, freq, theta, phi,
            center[0], center[1], center[2]
        )
        return nf2ff_result, freq



    def plot_radiation_pattern(self, nf2ff_result, freq):
        theta = np.arange(-180, 181, 2)
        E_norm = 20 * np.log10(nf2ff_result.E_norm[0] / np.max(nf2ff_result.E_norm[0])) \
                 + 10 * np.log10(nf2ff_result.Dmax[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=E_norm[:, 0], theta=theta, name='XZ Plane'))
        fig.add_trace(go.Scatterpolar(r=E_norm[:, 1], theta=theta, name='YZ Plane'))
        fig.update_layout(title=f"Radiation Pattern @ {freq/1e9:.2f} GHz",
                          polar=dict(radialaxis=dict(visible=True, range=[-30, max(E_norm.max(), 5)])),
                          template="plotly_white")
        fig.show()

    def plot_3d_radiation(self, nf2ff_result):
        """Plot 3D radiation pattern (in dB scale) using Plotly."""
        theta = np.deg2rad(np.arange(-180, 181, 2))
        phi = np.deg2rad(np.array([0, 90]))
        E_norm = nf2ff_result.E_norm[0]

        # 3D coordinates from spherical to cartesian
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        r = 10 ** (E_norm.T / 20)  # Convert dB to linear magnitude

        X = r * np.cos(theta_grid) * np.sin(phi_grid)
        Y = r * np.sin(theta_grid) * np.sin(phi_grid)
        Z = r * np.cos(phi_grid)

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=True)])
        fig.update_layout(title="3D Radiation Pattern",
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                          template="plotly_white")
        fig.show()
