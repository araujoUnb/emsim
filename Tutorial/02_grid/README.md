# Avaliação de eficiência: malha uniforme vs não uniforme

Esta pasta contém scripts e resultados para avaliar **ganhos de eficiência** ao usar malha não uniforme (stretched grid) em vez de malha uniforme no FDTD.

## Objetivo

- Comparar **número de células**, **tempo de execução** e **células/segundo** para o mesmo problema físico:
  - **Uniforme:** resolução fina em todo o domínio.
  - **Não uniforme:** refinamento só na região de interesse; resto com células maiores.

## Como executar

A partir da raiz do projeto:

```bash
python Tutorial/02_grid/run_efficiency_benchmark.py
```

O script escreve:
- **results.md** – tabela com métricas (células, tempo, células/s) e conclusões.
- **results.csv** – mesmos dados em CSV para análise externa.
- **efficiency_plot.png** – gráfico de comparação (se matplotlib estiver disponível).

## Interpretação

- **Cenário com ganho:** domínio grande com uma região pequena que exige refinamento (ex.: descontinuidade, ressonador). A malha não uniforme reduz o total de células e o tempo.
- **Cenário sem ganho:** quando a maior parte do domínio precisa de resolução fina, uniforme e não uniforme tendem a ter contagem e tempo semelhantes; o overhead dos coeficientes variáveis pode ser marginal.

Resumo das conclusões está também em [docs/performance.md](../../docs/performance.md).
