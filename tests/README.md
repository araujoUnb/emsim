# Testes do emsim

Sistema completo de testes para o simulador FDTD **emsim**, organizado em categorias para facilitar desenvolvimento, validação e manutenção.

## 📁 Estrutura

```
tests/
├── unit/                    # Testes unitários (funções isoladas, rápidos)
│   ├── test_constants.py
│   ├── test_grid.py
│   ├── test_fields.py
│   ├── test_materials.py
│   ├── test_gaussian_pulse.py
│   ├── test_cpml.py
│   ├── test_pec.py
│   ├── test_modes.py
│   ├── test_port.py
│   ├── test_lumped_port.py
│   ├── test_patch_antenna.py
│   └── test_nf2ff.py
│
├── integration/             # Testes de integração (módulos combinados)
│   ├── test_solver_sources.py
│   ├── test_solver_boundaries.py
│   └── test_simulation_workflow.py
│
├── validation/              # Validação física (vs soluções analíticas)
│   ├── test_waveguide_modes.py
│   ├── test_cpml_absorption.py
│   ├── test_energy_conservation.py
│   ├── test_free_space_propagation.py
│   ├── test_pec_reflection.py
│   └── test_cavity_resonance.py
│
├── performance/             # Benchmarks de velocidade e memória
│   └── (a implementar)
│
├── conftest.py              # Fixtures compartilhados (pytest)
├── run_all_tests.py         # Script principal para executar testes
└── README.md                # Este arquivo
```

## 🚀 Execução Rápida

**⚠️ ATENÇÃO**: Ative o `.venv` antes de executar os testes!

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Depois execute os testes:
python tests/run_all_tests.py
```

### Todos os testes
```bash
python tests/run_all_tests.py
```

### Por categoria
```bash
# Testes unitários (~10-20s)
python tests/run_all_tests.py --unit

# Testes de integração (~30-60s)
python tests/run_all_tests.py --integration

# Testes de validação física (~2-5min)
python tests/run_all_tests.py --validation

# Benchmarks de performance
python tests/run_all_tests.py --performance
```

### Modo de desenvolvimento
```bash
# Testes rápidos para desenvolvimento (unit + integration)
python tests/run_all_tests.py --fast

# Com output detalhado
python tests/run_all_tests.py --fast --verbose

# Com cobertura de código
python tests/run_all_tests.py --unit --coverage
```

## 🔍 Usando pytest Diretamente

```bash
# Todos os testes
pytest tests/

# Apenas unitários
pytest tests/unit/

# Apenas validação (verbose)
pytest tests/validation/ -v

# Testes específicos por nome
pytest tests/ -k "waveguide"

# Com cobertura
pytest tests/ --cov=emsim --cov-report=html

# Executar apenas testes marcados como "slow"
pytest tests/ -m "slow"

# Executar tudo EXCETO testes lentos
pytest tests/ -m "not slow"
```

## 📊 Categorias de Teste

### 1. **Unit Tests** (`unit/`)
Testes unitários de funções e classes individuais. Rápidos, sem dependências externas.

**Objetivo**: Verificar que cada componente funciona isoladamente.

**Exemplos**:
- `test_grid.py`: Criação de `YeeGrid`, espaçamento dx/dy/dz
- `test_cpml.py`: Perfis de sigma/kappa da CPML
- `test_lumped_port.py`: Injeção/gravação de porta lumped

**Tempo**: ~10-20 segundos

### 2. **Integration Tests** (`integration/`)
Testes de múltiplos módulos trabalhando juntos. Verificam que as interfaces entre componentes funcionam corretamente.

**Objetivo**: Detectar problemas de integração entre módulos.

**Exemplos**:
- `test_solver_sources.py`: Solver + fontes (Gaussian, lumped port)
- `test_solver_boundaries.py`: Solver + CPML + PEC
- `test_simulation_workflow.py`: YAML → Simulation → resultados

**Tempo**: ~30-60 segundos

### 3. **Validation Tests** (`validation/`)
Testes de validação física que comparam resultados numéricos com soluções analíticas ou leis físicas fundamentais.

**Objetivo**: Garantir correção física do simulador.

**Exemplos**:
- `test_waveguide_modes.py`: Frequência de corte TE10 vs c/(2a)
- `test_cpml_absorption.py`: Reflexão < -40 dB
- `test_energy_conservation.py`: Conservação de energia em meio sem perdas
- `test_free_space_propagation.py`: Velocidade da luz c = 3×10⁸ m/s
- `test_pec_reflection.py`: Coeficiente de reflexão Γ = -1
- `test_cavity_resonance.py`: Frequências ressonantes de cavidade

**Tolerâncias típicas**:
- Frequências: ±0.5%
- Impedâncias: ±2%
- Absorção CPML: < -40 dB (ou -30 dB relaxado)
- Conservação energia: < 5%

**Tempo**: ~2-5 minutos

### 4. **Performance Tests** (`performance/`)
Benchmarks de velocidade, memória e escalabilidade.

**Objetivo**: Detectar regressões de performance e otimizar gargalos.

**Tempo**: Variável (pode ser longo)

## 🛠️ Fixtures Compartilhados

O arquivo `conftest.py` fornece fixtures reutilizáveis:

```python
@pytest.fixture
def small_grid():
    """Grid pequeno para testes rápidos."""
    return YeeGrid(x_range=(0, 10.7e-3), y_range=(0, 4.3e-3), 
                   z_range=(0, 20e-3), f0=10e9, resolution=10)

@pytest.fixture
def gaussian_source():
    """Fonte gaussiana padrão."""
    return GaussianPulse(f0=10e9, bandwidth=5e9)

@pytest.fixture
def analytical_solutions():
    """Soluções analíticas para validação."""
    return {
        'te10_cutoff': lambda a: C0 / (2*a),
        'cavity_frequency': lambda m,n,p,a,b,d: ...,
        ...
    }
```

Uso nos testes:

```python
def test_my_feature(small_grid, analytical_solutions):
    fc = analytical_solutions['te10_cutoff'](10.7e-3)
    assert fc == pytest.approx(14.05e9, rel=0.01)
```

## 📝 Markers (Pytest)

Markers personalizados definidos em `conftest.py`:

- `@pytest.mark.slow`: Testes demorados (validação física)
- `@pytest.mark.benchmark`: Benchmarks de performance
- `@pytest.mark.integration`: Testes de integração
- `@pytest.mark.validation`: Testes de validação física

Uso:

```python
@pytest.mark.validation
@pytest.mark.slow
def test_cavity_modes():
    # Teste lento de validação
    pass
```

Executar apenas testes rápidos:

```bash
pytest tests/ -m "not slow"
```

## ✅ Critérios de Sucesso

Para que o simulador seja considerado validado:

1. ✅ **Cobertura de código > 80%**
   ```bash
   pytest tests/ --cov=emsim --cov-report=term
   ```

2. ✅ **Todos os testes unitários passam** (< 30s)
   ```bash
   pytest tests/unit/
   ```

3. ✅ **Todos os testes de integração passam** (< 1min)
   ```bash
   pytest tests/integration/
   ```

4. ✅ **Todos os testes de validação passam** (< 5min)
   ```bash
   pytest tests/validation/
   ```

5. ✅ **Zero falhas em CI/CD** (a configurar)

## 🐛 Debugging Testes

### Ver output completo
```bash
pytest tests/unit/test_grid.py -v -s
# -v: verbose
# -s: não capturar print()
```

### Executar um único teste
```bash
pytest tests/unit/test_grid.py::test_grid_creation
```

### Parar no primeiro erro
```bash
pytest tests/ -x
```

### Debugger interativo
```bash
pytest tests/unit/test_grid.py --pdb
# Para quando houver falha e abre debugger
```

### Ver testes disponíveis sem executar
```bash
pytest tests/ --collect-only
```

## 📦 Dependências

**IMPORTANTE**: O projeto usa um ambiente virtual Python (`.venv`). Siga os passos:

### 1. Ativar o ambiente virtual

**Windows (PowerShell)**:
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD)**:
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac**:
```bash
source .venv/bin/activate
```

### 2. Instalar dependências de teste

```bash
pip install pytest pytest-cov pytest-benchmark scipy
```

Ou via `pyproject.toml` (se configurado):

```bash
pip install -e ".[test]"
```

### 3. Verificar instalação
```bash
pytest --version
# Deve mostrar: pytest 7.x.x ou superior
```

**Nota**: Todos os comandos neste README assumem que o `.venv` está ativo!

## 🔄 Workflow de Desenvolvimento

1. **Escrevendo código novo**:
   ```bash
   # Execute unit tests frequentemente
   pytest tests/unit/ -k "minha_feature"
   ```

2. **Antes de commit**:
   ```bash
   # Testes rápidos + linting
   pytest tests/ --fast
   ```

3. **Antes de merge/PR**:
   ```bash
   # Validação completa
   pytest tests/ --cov=emsim --cov-report=html
   ```

4. **Após mudanças críticas**:
   ```bash
   # Validação física completa
   pytest tests/validation/ -v
   ```

## 📚 Adicionando Novos Testes

### Teste unitário
1. Crie arquivo em `tests/unit/test_meu_modulo.py`
2. Use fixtures de `conftest.py`
3. Teste comportamento isolado

```python
def test_minha_funcao(small_grid):
    result = minha_funcao(small_grid)
    assert result > 0
```

### Teste de validação
1. Crie arquivo em `tests/validation/test_meu_fenomeno_fisico.py`
2. Compare com solução analítica ou lei física
3. Marque como `@pytest.mark.validation` e `@pytest.mark.slow`

```python
@pytest.mark.validation
@pytest.mark.slow
def test_meu_fenomeno(analytical_solutions):
    valor_simulado = simular()
    valor_analitico = analytical_solutions['formula'](params)
    assert valor_simulado == pytest.approx(valor_analitico, rel=0.01)
```

## 📈 Métricas de Qualidade

Execute para verificar métricas:

```bash
# Cobertura detalhada
pytest tests/ --cov=emsim --cov-report=term-missing

# Performance (cells/segundo)
pytest tests/performance/ --benchmark-only

# Contagem de testes
pytest tests/ --collect-only | grep "test session starts"
```

## 🚨 Troubleshooting

### Testes falhando após mudanças
1. Execute apenas o teste que falhou: `pytest tests/unit/test_x.py::test_y -v`
2. Verifique se a mudança quebrou a interface
3. Atualize os testes se o comportamento mudou intencionalmente

### Testes lentos
1. Marque como `@pytest.mark.slow`
2. Execute testes rápidos com: `pytest -m "not slow"`
3. Considere reduzir `n_steps` ou `resolution`

### Testes inconsistentes
1. Verifique dependências entre testes (use fixtures)
2. Garanta reset de estado global
3. Use seeds para aleatoriedade: `np.random.seed(42)`

## 📞 Suporte

- **Issues**: Abra uma issue no repositório
- **Documentação**: Veja docstrings em cada teste
- **Exemplos**: Veja testes existentes como referência

---

**Última atualização**: 2026-02-14  
**Versão**: 1.0  
**Autor**: emsim development team
