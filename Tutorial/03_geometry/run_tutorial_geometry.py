#!/usr/bin/env python3
"""
Executa o tutorial de geometria sem Jupyter: formas básicas, operações e waveguide.
Gera figuras em Tutorial/03_geometry/figures/.
Modo interativo (janela que pode rotacionar): use --interactive.
  python Tutorial/03_geometry/run_tutorial_geometry.py --interactive
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main(interactive: bool = False):
    from emsim.geometry import (
        Box,
        Cylinder,
        Sphere,
        RectangularWaveguide,
        plot_geometry,
    )

    print("1. Formas básicas")
    caixa = Box(0.0, 0.01, 0.0, 0.005, 0.0, 0.02)
    cilindro = Cylinder(center_x=0.005, center_y=0.0025, radius=0.002, z_min=0.0, z_max=0.02)
    esfera = Sphere(center_x=0.005, center_y=0.0025, center_z=0.01, radius=0.003)
    print("   Box bounds:", caixa.bounds())
    print("   Cylinder bounds:", cilindro.bounds())
    print("   Sphere bounds:", esfera.bounds())

    print("\n2. Visualizações")
    if interactive:
        print("   Modo interativo: feche cada janela para passar à próxima (pode rotacionar com o rato).")
        plot_geometry(caixa, backend="pyvista", notebook=False)
        plot_geometry(cilindro, backend="pyvista", notebook=False)
        plot_geometry(esfera, backend="pyvista", notebook=False)
    else:
        plot_geometry(caixa, backend="matplotlib", save_path=str(FIG_DIR / "box.png"))
        plot_geometry(cilindro, backend="matplotlib", save_path=str(FIG_DIR / "cylinder.png"))
        plot_geometry(esfera, backend="matplotlib", save_path=str(FIG_DIR / "sphere.png"))
        print("   Guardado: box.png, cylinder.png, sphere.png")

    print("\n3. Operação booleana: caixa - cilindro (caixa com furo)")
    try:
        import pyvista as pv
        box_mesh = caixa.to_pyvista().triangulate()
        cyl_mesh = cilindro.to_pyvista().triangulate()
        caixa_com_furo = box_mesh.boolean_difference(cyl_mesh)
        # Semi-transparent so the cylindrical hole is visible; view from above-front so the circular opening is clear.
        cx, cy, cz = 0.005, 0.0025, 0.01
        d = 0.035
        camera_pos = [(cx + d, cy + d * 0.6, cz + d), (cx, cy, cz), (0, 1, 0)]
        if interactive:
            pl = pv.Plotter()
            pl.add_mesh(caixa_com_furo, show_edges=True, color="tan", opacity=0.55)
            pl.camera_position = camera_pos
            pl.show()
        else:
            pl = pv.Plotter(off_screen=True)
            pl.add_mesh(caixa_com_furo, show_edges=True, color="tan", opacity=0.55)
            pl.camera_position = camera_pos
            pl.screenshot(str(FIG_DIR / "box_minus_cylinder.png"))
            # Corte por um plano (x = cx) para mostrar o furo cilíndrico em secção
            clipped = caixa_com_furo.clip(normal="x", origin=(cx, 0, 0))
            pl2 = pv.Plotter(off_screen=True)
            pl2.add_mesh(clipped, show_edges=True, color="tan", opacity=0.7)
            pl2.view_xy()
            pl2.screenshot(str(FIG_DIR / "box_minus_cylinder_section.png"))
            pl2.close()
            pl.close()
            print("   Guardado: box_minus_cylinder.png, box_minus_cylinder_section.png (corte para ver o furo)")
    except Exception as e:
        print("   Aviso: não foi possível gerar box_minus_cylinder.png:", e)

    print("\n4. Waveguide rectangular (WR-42)")
    a, b, length = 10.67e-3, 4.32e-3, 50e-3
    waveguide = RectangularWaveguide(a=a, b=b, length=length)
    print("   x_range:", waveguide.x_range)
    print("   y_range:", waveguide.y_range)
    print("   z_range:", waveguide.z_range)
    if interactive:
        print("   Waveguide com volume semi-transparente (no modelo FDTD as paredes são PEC, sem espessura).")
        plot_geometry(waveguide, backend="pyvista", notebook=False)
    else:
        plot_geometry(waveguide, backend="matplotlib", save_path=str(FIG_DIR / "waveguide.png"))
        print("   Guardado: waveguide.png")

    if not interactive:
        print("\nConcluído. Figuras em:", FIG_DIR)
    print("   Para rotacionar: use --interactive ou abra o notebook no Jupyter (backend=\"pyvista\").")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tutorial de geometria: gera figuras ou abre janelas interactivas.")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Abre uma janela PyVista por geometria para poder rotacionar (feche cada janela para continuar).",
    )
    args = parser.parse_args()
    sys.exit(main(interactive=args.interactive))
