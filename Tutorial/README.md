# Tutorial EMSIM: electromagnetismo e simulador FDTD

Este tutorial organiza o estudo dos **aspectos fundamentais** do electromagnetismo e do simulador FDTD em subpastas numeradas. Cada subpasta cobre um tema e, quando há fórmulas analíticas, inclui comparação entre **curvas teóricas** e resultados do simulador.

## Percurso recomendado

Siga a ordem **01 → 08** para um percurso didático completo (do método FDTD e da malha até à análise de resultados).

| # | Pasta | Tema |
|---|-------|------|
| 01 | [01_fundamentals](01_fundamentals/) | FDTD, malha de Yee, condição CFL; propagação em espaço livre. Comparação: velocidade da onda vs c₀, opcionalmente impedância vs η₀. |
| 02 | [02_grid](02_grid/) | Malha uniforme vs não uniforme (stretched), resolução, eficiência (células, tempo). |
| 03 | [03_geometry](03_geometry/) | Geometria: formas básicas (Box, Cylinder, Sphere), operações booleanas, waveguide rectangular; visualização 3D. |
| 04 | [04_boundaries](04_boundaries/) | Condições de fronteira: PEC (reflexão total) e CPML (absorbing). Comparação: Γ PEC vs -1; potência reflectida com CPML. |
| 05 | [05_materials](05_materials/) | Materiais: vácuo, dieléctricos, dispersivos (Drude/Lorentz), anisotrópicos. Comparação: ε(f) e resposta em frequência. |
| 06 | [06_waveguides_modes](06_waveguides_modes/) | Guias rectangulares: modos TE/TM, frequência de corte, ressonância em cavidade. Comparação: fc, f_mnp, Z_TE vs teoria. |
| 07 | [07_sources_ports](07_sources_ports/) | Fontes (Gaussian pulse) e portos; injeção de campos. Comparação: forma do pulso no tempo (e opcionalmente espectro). |
| 08 | [08_postprocessing](08_postprocessing/) | Resultados: S-parameters, snapshots de campos, padrão de radiação. Comparação: S11/S21 e padrão vs teoria em casos simples. |

## Como usar

- Execute os scripts ou abra os notebooks em cada subpasta a partir da **raiz do projeto** (para que `emsim` seja importável).
- Em pastas com “teoria vs simulação”, os exemplos mostram gráficos ou tabelas que sobrepõem a curva teórica e o resultado do simulador.

## Dependências

- `emsim` (instalado no ambiente do projeto)
- `pyvista`, `matplotlib`, `numpy`, `tensorflow` (conforme os exemplos)
