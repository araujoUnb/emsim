#!/usr/bin/env python
r"""Execute todos os testes do emsim com relatório detalhado.

IMPORTANTE: Ative o ambiente virtual (.venv) antes de executar!
    Windows: .\.venv\Scripts\Activate.ps1
    Linux/Mac: source .venv/bin/activate

Usage:
    python tests/run_all_tests.py                 # Todos os testes
    python tests/run_all_tests.py --unit          # Apenas unitários
    python tests/run_all_tests.py --integration   # Apenas integração
    python tests/run_all_tests.py --validation    # Apenas validação física
    python tests/run_all_tests.py --fast          # Testes rápidos (unit + integration)
    python tests/run_all_tests.py --verbose       # Output detalhado
    python tests/run_all_tests.py --coverage      # Com relatório de cobertura

Examples:
    # Desenvolvimento rápido (unit tests apenas)
    python tests/run_all_tests.py --unit

    # Antes de commit (unit + integration, rápido)
    python tests/run_all_tests.py --fast

    # Validação completa (lento, inclui testes físicos)
    python tests/run_all_tests.py --validation -v

    # Tudo com cobertura
    python tests/run_all_tests.py --coverage -v
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_tests(test_dir: str = None, markers: list = None, verbose: bool = False, 
              coverage: bool = False):
    """Execute testes em um diretório específico com pytest.
    
    Args:
        test_dir: Diretório de testes (None = todos)
        markers: Lista de markers pytest (e.g., ['slow', 'integration'])
        verbose: Se True, usa -v para output detalhado
        coverage: Se True, gera relatório de cobertura
    
    Returns:
        Código de saída do pytest (0 = sucesso)
    """
    tests_root = Path(__file__).parent
    
    if test_dir:
        target = str(tests_root / test_dir)
    else:
        target = str(tests_root)
    
    cmd = ["pytest", target]
    
    # Markers
    if markers:
        cmd.extend(["-m", " or ".join(markers)])
    
    # Verbose
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")  # Quiet para output mais limpo
    
    # Coverage
    if coverage:
        cmd.extend([
            "--cov=emsim",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    
    # Opções adicionais
    cmd.extend([
        "--tb=short",      # Traceback curto
        "-ra",             # Resumo de todos os resultados
        "--strict-markers", # Erro se marker não definido
    ])
    
    print(f"\n{'='*80}")
    print(f"Executando: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=tests_root.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Execute testes do emsim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Categorias de teste
    parser.add_argument("--unit", action="store_true", 
                       help="Apenas testes unitários (~10s)")
    parser.add_argument("--integration", action="store_true",
                       help="Apenas testes de integração (~30s)")
    parser.add_argument("--validation", action="store_true",
                       help="Apenas testes de validação física (~2-5min)")
    parser.add_argument("--performance", action="store_true",
                       help="Apenas benchmarks de performance")
    parser.add_argument("--fast", action="store_true",
                       help="Testes rápidos: unit + integration (~40s)")
    
    # Opções de output
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Output verbose com detalhes de cada teste")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Gerar relatório de cobertura de código")
    
    # Opções de filtragem
    parser.add_argument("--keyword", "-k", type=str,
                       help="Executar apenas testes que contêm palavra-chave")
    parser.add_argument("--marker", "-m", type=str,
                       help="Executar apenas testes com marker específico")
    
    args = parser.parse_args()
    
    # Determinar qual conjunto de testes executar
    exit_code = 0
    
    if args.unit:
        print("🧪 Executando testes unitários...")
        exit_code |= run_tests("unit", verbose=args.verbose, coverage=args.coverage)
    
    elif args.integration:
        print("🔗 Executando testes de integração...")
        exit_code |= run_tests("integration", verbose=args.verbose, coverage=args.coverage)
    
    elif args.validation:
        print("✓ Executando testes de validação física (lento)...")
        exit_code |= run_tests("validation", verbose=args.verbose, coverage=args.coverage)
    
    elif args.performance:
        print("⚡ Executando benchmarks de performance...")
        exit_code |= run_tests("performance", markers=["benchmark"], 
                              verbose=args.verbose, coverage=False)
    
    elif args.fast:
        print("⚡ Modo rápido: unit + integration...")
        print("\n--- Unit Tests ---")
        exit_code |= run_tests("unit", verbose=args.verbose, coverage=args.coverage)
        
        if exit_code == 0:  # Só continua se unit passou
            print("\n--- Integration Tests ---")
            exit_code |= run_tests("integration", verbose=args.verbose, coverage=False)
    
    else:
        # Todos os testes
        print("🚀 Executando TODOS os testes (unit + integration + validation)...")
        exit_code |= run_tests(None, verbose=args.verbose, coverage=args.coverage)
    
    # Resultado final
    print(f"\n{'='*80}")
    if exit_code == 0:
        print("✅ SUCESSO: Todos os testes passaram!")
    else:
        print("❌ FALHA: Alguns testes falharam.")
    print(f"{'='*80}\n")
    
    # Lembrete sobre cobertura
    if args.coverage:
        print("📊 Relatório de cobertura gerado em: htmlcov/index.html")
        print("   Abra no navegador para visualizar.\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
