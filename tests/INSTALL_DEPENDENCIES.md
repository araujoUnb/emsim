# Instalação de Dependências de Teste - emsim

Este guia ajuda a configurar o ambiente de testes do emsim.

## Passo 1: Ativar o Ambiente Virtual

O projeto usa `.venv` como ambiente virtual Python.

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

Se houver erro de política de execução:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Windows CMD
```cmd
.venv\Scripts\activate.bat
```

### Linux/Mac
```bash
source .venv/bin/activate
```

## Passo 2: Instalar Dependências de Teste

Com o `.venv` ativo, instale as bibliotecas necessárias:

```bash
pip install pytest pytest-cov pytest-benchmark scipy
```

## Passo 3: Verificar Instalação

```bash
pytest --version
```

Deve exibir algo como: `pytest 7.4.x`

## Passo 4: Executar Testes

```bash
# Testes unitários (rápido)
python tests/run_all_tests.py --unit

# Todos os testes
python tests/run_all_tests.py
```

## Solução de Problemas

### "pytest: command not found" ou "No module named pytest"

**Causa**: O `.venv` não está ativo ou pytest não foi instalado.

**Solução**:
1. Certifique-se que o `.venv` está ativo (deve aparecer `(.venv)` no prompt)
2. Reinstale: `pip install pytest`

### "ModuleNotFoundError: No module named 'emsim'"

**Causa**: O pacote `emsim` não está no path do Python.

**Solução**:
```bash
# Na raiz do projeto (com .venv ativo)
pip install -e .
```

### Testes falhando com "ImportError"

**Causa**: Falta alguma dependência.

**Solução**:
```bash
pip install scipy numpy tensorflow pyyaml matplotlib
```

## Dependências Completas

```
# Core (já deve estar instalado)
numpy
tensorflow
pyyaml
matplotlib

# Testes
pytest>=7.0
pytest-cov>=4.0
pytest-benchmark>=4.0
scipy>=1.9
```

## Próximos Passos

Após instalação bem-sucedida:

1. Execute testes unitários: `python tests/run_all_tests.py --unit`
2. Se passar, execute testes completos: `python tests/run_all_tests.py --fast`
3. Leia `tests/README.md` para documentação completa

---

**Última atualização**: 2026-02-14
