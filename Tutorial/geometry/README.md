# Tutorial: Geometria, formas básicas e visualização

Este tutorial mostra como usar o pacote `emsim.geometry` para:

1. **Formas básicas** — Box, Cylinder, Sphere (primitivas com `bounds()` e `to_pyvista()`).
2. **Visualizações** — `plot_geometry()` com PyVista ou matplotlib (Jupyter e export para PNG).
3. **Operações** — subtracção booleana entre formas (ex.: caixa com furo) usando PyVista.
4. **Construir um waveguide** — definição de um guia rectangular e ligação à simulação FDTD.

## Como executar

Recomendado: abrir o notebook no Jupyter para ver os gráficos 3D interativos.

```bash
cd /path/to/emsim
jupyter notebook Tutorial/geometry/tutorial_geometry.ipynb
```

Ou executar como script (gera imagens em `Tutorial/geometry/figures/`):

```bash
python Tutorial/geometry/run_tutorial_geometry.py
```

**Ver volume e rotacionar a cena 3D**

- **Janela (desktop):** use o modo interativo para abrir uma janela PyVista por geometria; pode rotacionar com o rato.
  ```bash
  python Tutorial/geometry/run_tutorial_geometry.py --interactive
  ```
- **No browser (Jupyter):** no notebook, use `plot_geometry(..., backend="pyvista")` para ver o widget 3D no próprio notebook e rotacionar aí.

No PyVista, o **waveguide** e a **caixa** são desenhados com superfície semi-transparente (`opacity=0.45`) para se ver o interior e a sensação de volume; no modelo FDTD as paredes do waveguide continuam a ser PEC (espessura nula).

## Conteúdo do tutorial

1. **Formas básicas** — Box, Cylinder, Sphere (`bounds()`, `to_pyvista()`).
2. **Visualizações** — `plot_geometry()` com PyVista (Jupyter) ou matplotlib (export para PNG).
3. **Operações** — subtracção booleana (caixa − cilindro = caixa com furo) via `pyvista` (meshes triangulados).
4. **Waveguide** — construção de um `RectangularWaveguide` (ex.: WR-42) e exemplo de config para a simulação.

O script `run_tutorial_geometry.py` gera em `figures/`: `box.png`, `cylinder.png`, `sphere.png`, `box_minus_cylinder.png`, `box_minus_cylinder_section.png` (corte que mostra o furo cilíndrico), `waveguide.png`. Para ver o furo da caixa com mais clareza, use a figura de secção ou `--interactive` e rotacione a cena.

## Dependências

- `emsim` (instalado no ambiente do projeto)
- `pyvista` — visualização 3D e operações booleanas
- `matplotlib` — fallback e export
