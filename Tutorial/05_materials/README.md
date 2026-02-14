# 05 — Materiais

Vácuo, **dieléctricos** (εr), **condutores** (σ), materiais **dispersivos** (Drude, Lorentz, Debye) e **anisotrópicos** (tensores ε, μ).

## O que vai aprender

- Como definir materiais no simulador (catálogo, CSV/JSON).
- Comportamento em frequência de materiais dispersivos (ε(f)).
- Uso de materiais anisotrópicos quando a permissividade depende da direcção.

## Comparação teoria vs simulação

- **Dispersivos:** ε(f) analítico (Drude/Lorentz) vs resposta em frequência da simulação.
- **Anisotrópicos:** índice efectivo ou reflexão vs teoria, quando aplicável.

## Base no projeto

- [emsim/materials/](../../emsim/materials/) — Material, DispersiveMaterial, AnisotropicMaterial, MaterialManager, catálogo.
- [tests/validation/test_dispersive_physics.py](../../tests/validation/test_dispersive_physics.py), [test_anisotropic_materials.py](../../tests/validation/test_anisotropic_materials.py).

## Conteúdo

- **dielectric.ipynb** — Propagação em vácuo e em dieléctrico (eps_r=2); medição da velocidade; comparação v_diel/v_vac vs 1/sqrt(eps_r).

## Como executar

A partir da raiz do projeto:

```bash
jupyter notebook Tutorial/05_materials/dielectric.ipynb
```

## Pré-requisitos

Recomendado: 01_fundamentals, 02_grid, 04_boundaries.
