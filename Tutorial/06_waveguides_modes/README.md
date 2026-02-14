# 06 — Guias de onda e modos

Modos **TE** e **TM** em guia rectangular; **frequência de corte**; **ressonância** em cavidade rectangular com paredes PEC.

## O que vai aprender

- Frequência de corte TE10: fc = c/(2a) e propagação acima do corte.
- Fórmula de ressonância da cavidade: f_mnp.
- Impedância do modo TE em função da frequência, Z_TE(f).

## Comparação teoria vs simulação

Gráficos teoria vs sim são especialmente didáticos aqui:

- **Frequência de corte TE10:** fc = c/(2a) vs pico S11 ou dispersão da simulação.
- **Cavidade:** f_mnp analítico vs picos de ressonância simulados.
- **Impedância TE:** Z_TE(f) teórica vs extraída da simulação.

## Base no projeto

- [emsim/modes/rectangular.py](../../emsim/modes/rectangular.py) — cutoff_frequency, te_mode_profile, mode_impedance.
- [tests/validation/test_waveguide_modes.py](../../tests/validation/test_waveguide_modes.py), [test_cavity_resonance.py](../../tests/validation/test_cavity_resonance.py).
- [tests/conftest.py](../../tests/conftest.py) — fixture analytical_solutions (te10_cutoff, cavity_frequency, te_impedance).

## Conteúdo

- **cutoff_and_cavity.ipynb** — Cavidade: FFT do campo no centro vs frequências teóricas f_mnp; guia: fc TE10 e Z_TE(f) teóricos (WR42).

## Como executar

A partir da raiz do projeto:

```bash
jupyter notebook Tutorial/06_waveguides_modes/cutoff_and_cavity.ipynb
```

## Pré-requisitos

Recomendado: 01_fundamentals a 05_materials.
