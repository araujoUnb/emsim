# 08 — Pós-processamento e visualização

**S-parameters** (S11, S21), **snapshots** de campos (Ez, etc.) no tempo, **padrão de radiação** e transformada **NF2FF** (near-to-far field).

## O que vai aprender

- Como obter e representar S11/S21 a partir dos resultados da simulação.
- Visualização de campos 2D/3D em instantes de tempo.
- Cálculo do diagrama de radiação quando aplicável.

## Comparação teoria vs simulação

- **S11/S21** vs teoria em casos simples (ex.: guia com carga conhecida, cavidade).
- **Padrão de radiação** vs fórmula analítica quando existir (ex.: dipolo).

## Base no projeto

- [emsim/postprocessing/](../../emsim/postprocessing/) — plot_s_parameters, plot_field_snapshots, plot_structure_3d, radiation_pattern, nf2ff.
- [Simulations/WR42/postprocess.py](../../Simulations/WR42/postprocess.py) — exemplo de script de pós-processamento.

## Conteúdo

- **plot_s_params_with_theory.py** — Carrega s_parameters.csv (ex.: de Simulations/WR42/outputs), plota |S11| e |S21| em dB e sobrepõe a frequência de corte TE10 (teoria). Gera `figures/s_params_with_fc_te10.png`.

Requer ter corrido uma simulação que produza s_parameters.csv (ex.: `python Simulations/WR42/run.py`).

## Como executar

A partir da raiz do projeto:

```bash
python Tutorial/08_postprocessing/plot_s_params_with_theory.py [output_dir]
```

Se omitir output_dir, usa Simulations/WR42/outputs.

## Pré-requisitos

Recomendado: tópicos 01 a 07; dados de simulação (ex.: WR42).
