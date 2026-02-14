# 04 — Condições de fronteira

**PEC** (Perfect Electric Conductor): paredes metálicas que reflectem a onda (campo tangencial E = 0). **CPML** (Convolutional PML): camadas absorventes para simular domínio infinito.

## O que vai aprender

- Como a fronteira PEC impõe reflexão total e coeficiente de reflexão Gamma = -1.
- Como o CPML reduz reflexões parasitas nas bordas do domínio.

## Comparação teoria vs simulação

- **Reflexão PEC:** coeficiente Gamma = -1 (teoria) vs valor inferido da simulação (amplitude reflectida / incidente).
- **CPML:** potência reflectida vs teoria (idealmente baixa).

## Base no projeto

- emsim/boundaries/ — pec.py, cpml.py.
- tests/validation/test_pec_reflection.py, test_cpml_absorption.py.

## Conteúdo

- **pec_reflection.py** — Onda a propagar em direcção a PEC (z=0); grava Ey(t) em dois pontos; gera figura `figures/pec_reflection_ey_t.png`. Comparação: teoria Gamma = -1.
- **cpml_absorption.py** — Malha com CPML em z- e z+; plota |Ey| ao longo de z mostrando decaimento nas PML. Gera `figures/cpml_absorption_ey_z.png`.

## Como executar

A partir da raiz do projeto:

```bash
python Tutorial/04_boundaries/pec_reflection.py
python Tutorial/04_boundaries/cpml_absorption.py
```

## Pré-requisitos

Recomendado: ter feito 01_fundamentals e 02_grid.
