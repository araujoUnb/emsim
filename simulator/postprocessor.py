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
        idx = np.argmin(s11_dB)
        freq = self.f[idx]
        
        print(f"📡 Calculando NF2FF em {freq/1e9:.2f} GHz")

        nf2ff_result = self.nf2ff_box.CalcNF2FF(
            self.sim_path, freq, theta, phi,
            center=center
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

  

    def plot_3d_radiation(self,nf2ff_result):
        """Plota o padrão de radiação 3D em dB usando Plotly."""
        # Extraindo os ângulos theta e phi do resultado NF2FF
        theta = np.deg2rad(nf2ff_result.theta)  # Convertendo para radianos
        phi = np.deg2rad(nf2ff_result.phi)      # Convertendo para radianos

        # Criando uma grade de coordenadas esféricas
        theta_grid, phi_grid = np.meshgrid(phi,theta)

        # Extraindo a magnitude normalizada do campo elétrico
        E_norm = nf2ff_result.E_norm

        # Convertendo a magnitude para escala logarítmica (dB)
        E_norm_dB = 20 * np.log10(abs(E_norm[0]))

        # Convertendo coordenadas esféricas para cartesianas para plotagem 3D
        r = E_norm_dB  # Usando os valores em dB diretamente para o raio
        X = r * np.sin(theta_grid) * np.cos(phi_grid)
        Y = r * np.sin(theta_grid) * np.sin(phi_grid)
        Z = r * np.cos(theta_grid)

        # Criando o gráfico de superfície 3D
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=E_norm_dB, colorscale='Viridis', showscale=True)])
        fig.update_layout(
            title="Padrão de Radiação 3D (dB)",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='auto'
            ),
            template="plotly_white"
        )
        fig.show()

