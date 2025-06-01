import torch
from utils.utils_rectangular_waveguide import TEModeProfile, TMModeProfile

# Defina dimensões e grades
a = 1.0
b = 0.5
Nx = 50
Ny = 40
x_lin = torch.linspace(0.0, a, Nx)
y_lin = torch.linspace(0.0, b, Ny)

# Instancie TE11
te11 = TEModeProfile(a=a, b=b, m=1, n=1, x_lin=x_lin, y_lin=y_lin, device='cpu')

# 1) Heatmap 2D
te11.plot_heatmap()

# 2) Corte central 1D (perfil vs x, em y ≈ b/2)
te11.plot_line(direction='x')

# 3) Corte central 1D (perfil vs y, em x ≈ a/2)
te11.plot_line(direction='y')

# 4) Surface 3D
te11.plot_3d()


# Instancie TM11
tm11 = TMModeProfile(a=a, b=b, m=1, n=1, x_lin=x_lin, y_lin=y_lin, device='cpu')

# Heatmap 2D
tm11.plot_heatmap()

# Corte vs x
tm11.plot_line(direction='x')

# Corte vs y
tm11.plot_line(direction='y')

# Surface 3D
tm11.plot_3d()
