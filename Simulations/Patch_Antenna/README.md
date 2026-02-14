# Patch Antenna Simulation (2.4 GHz)

Simulação de antena patch para 2.4 GHz usando o framework emsim, replicando o exemplo do openEMS.

## Descrição

Esta simulação implementa uma antena patch microstrip ressonante em 2.4 GHz sobre substrato FR-4:
- **Patch**: 32 mm × 40 mm (PEC)
- **Substrato**: FR-4 (εᵣ = 3.38, espessura = 1.524 mm)
- **Ground plane**: PEC no fundo do substrato
- **Alimentação**: Porta lumped em posição -6 mm (50 Ω)
- **Domínio**: Caixa de ar 200×200×150 mm com PML

## Características Implementadas

### Componentes Novos
1. **Geometria**: `PatchAntenna` class
2. **Porta Lumped**: Alimentação pontual com impedância de entrada
3. **Materiais Heterogêneos**: Substrato dielétrico + ar
4. **PEC Interno**: Patch e ground plane
5. **NF2FF**: Gravação de campos para padrão de radiação

### Solver Generalizado
O `FDTDSolver` foi estendido para suportar:
- Múltiplos tipos de portas (modal, lumped)
- Regiões PEC internas
- Gravação NF2FF opcional
- Interface comum via `PortBase` protocol

## Como Executar

### 1. Executar a Simulação

```bash
# Da raiz do projeto
python Simulations/Patch_Antenna/run.py

# Ou do diretório da simulação
cd Simulations/Patch_Antenna
python run.py
```

**Tempo estimado**: ~5-15 minutos (dependendo da GPU/CPU)

### 2. Gerar Gráficos

```bash
python Simulations/Patch_Antenna/postprocess.py
```

### 3. Resultados

Os resultados são salvos em `Simulations/Patch_Antenna/outputs/`:
- `s_parameters.csv` - Parâmetro S11 vs frequência
- `s11.png` - Gráfico de S11 (magnitude e fase)
- `ez_snapshots.csv` - Snapshots do campo Ez
- `fields.png` - Visualização dos campos
- `result_metadata.csv` - Metadados da simulação
- `run_info.yaml` - Log completo da execução

## Resultados Esperados

Com base no exemplo do openEMS:
- **Frequência de ressonância**: ~2.4 GHz (S11 < -10 dB)
- **Impedância de entrada**: ~50 Ω na ressonância
- **Padrão de radiação**: Lóbulo principal perpendicular ao patch
- **Diretividade**: ~6-8 dBi (típico para patch)

## Configuração

Edite `config.yaml` para ajustar parâmetros:

```yaml
# Resolução do grid (células por λ)
grid:
  resolution: 20  # Aumentar para mais precisão (mais lento)

# Número de passos temporais
run:
  n_steps: 30000  # Reduzir para teste rápido

# NF2FF (opcional)
nf2ff:
  enabled: true  # Desabilitar para simulação mais rápida
```

## Testes Rápidos

Para teste rápido (menor precisão):
```yaml
grid:
  resolution: 15
run:
  n_steps: 10000
nf2ff:
  enabled: false
```

Para alta precisão (mais lento):
```yaml
grid:
  resolution: 30
run:
  n_steps: 50000
```

## Arquitetura

```
Simulations/Patch_Antenna/
├── config.yaml          # Configuração da simulação
├── run.py              # Script de execução
├── postprocess.py      # Geração de gráficos
├── README.md           # Este arquivo
└── outputs/            # Resultados (criado automaticamente)
    ├── s_parameters.csv
    ├── s11.png
    ├── fields.png
    └── run_info.yaml
```

## Dependências

Todas as dependências estão no `pyproject.toml`:
- TensorFlow >= 2.15
- NumPy >= 1.24
- Pandas >= 2.0
- Matplotlib >= 3.7
- SciPy >= 1.10
- PyYAML >= 6.0

## Comparação com openEMS

| Aspecto | openEMS | emsim |
|---------|---------|-------|
| **Mesh** | Não-uniforme (refinado) | Uniforme (λ/20) |
| **Solver** | C++ compilado | TensorFlow (Python) |
| **GPU** | Não | Sim (automático) |
| **Porta** | Lumped port | LumpedPort class |
| **NF2FF** | Completo | Placeholder (em desenvolvimento) |

## Limitações Atuais

1. **NF2FF**: Transformação completa ainda não implementada (retorna estrutura placeholder)
2. **Mesh uniforme**: Menos eficiente que mesh adaptativo do openEMS
3. **Impedância**: Cálculo via FFT (pode requerer mais passos temporais para convergência)

## Próximos Passos

- [ ] Implementar transformação NF2FF completa
- [ ] Adicionar mesh não-uniforme
- [ ] Otimizar cálculo de impedância
- [ ] Validar com medições experimentais

## Referências

- openEMS Simple Patch Antenna Tutorial
- Balanis, C. A., "Antenna Theory: Analysis and Design"
- Taflove, A., "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
