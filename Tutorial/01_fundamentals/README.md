# 01 — Fundamentos FDTD

Introdução ao método **FDTD** (Finite-Difference Time-Domain), à **malha de Yee** e à condição **CFL** (Courant–Friedrichs–Lewy). Primeiro exemplo: propagação de uma onda em espaço livre.

## O que vai aprender

- O que é o FDTD e como as equações de Maxwell são discretizadas na malha de Yee.
- Por que a condição CFL limita o passo temporal.
- Como correr uma simulação mínima de propagação em vácuo.

## Comparação teoria vs simulação

Quando houver exemplos implementados: **velocidade da onda** medida na simulação vs c0 (teoria); opcionalmente **impedância do vácuo** vs eta0 aprox. 377 ohm.

## Base no projeto

- emsim/fdtd/grid.py — malha de Yee, CFL.
- tests/validation/test_free_space_propagation.py — validação de propagação em espaço livre.

## Como executar

Execute a partir da **raiz do projeto** para que `emsim` e `Tutorial.common` sejam importáveis:

```bash
cd /path/to/emsim
jupyter notebook Tutorial/01_fundamentals/propagation_free_space.ipynb
```

No primeiro cell, o notebook descobre a raiz do projeto (procurando a pasta `emsim`). Se abrir o notebook noutro diretório, certifique-se de que a raiz está em `sys.path` ou que inicia o Jupyter na raiz.

## Pré-requisitos

Nenhum (é o primeiro tópico do tutorial).
