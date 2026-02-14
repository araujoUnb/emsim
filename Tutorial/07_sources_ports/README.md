# 07 — Fontes e portos

**Gaussian pulse** (modulado em seno), **injeção** de campos na malha, **portos** (lumped, etc.) para excitação e extração de S-parameters.

## O que vai aprender

- Como definir uma fonte tipo Gaussian pulse (f0, largura de banda).
- Onde e como injectar E/H na malha.
- Uso de portos para alimentar estruturas (ex.: waveguide) e obter S11/S21.

## Comparação teoria vs simulação

- **Forma do pulso no tempo:** envelope Gaussian teórico vs campo injectado na simulação.
- Opcional: **espectro** da fonte vs teórico.

## Base no projeto

- [emsim/sources/](../../emsim/sources/) — GaussianPulse, injector.
- [emsim/ports/](../../emsim/ports/) — Port, lumped port.

## Conteúdo

- **gaussian_source.ipynb** — Fórmula do pulso Gaussian; comparação da forma teórica s(t) com o campo Ey no ponto de injeção na simulação.

## Como executar

A partir da raiz do projeto:

```bash
jupyter notebook Tutorial/07_sources_ports/gaussian_source.ipynb
```

## Pré-requisitos

Recomendado: 01_fundamentals a 06_waveguides_modes.
