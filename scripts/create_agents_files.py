#!/usr/bin/env python3
"""
Create AGENTS/ files for EMSIM project.
Writes: AGENTS/00_COORDINATOR.md, 01_EM_SPECIALIST.md, 02_SENIOR_DEV.md,
03_DEV_1.md, 04_DEV_2.md, 05_DEV_3.md
Run from project root (or adjust BASE_DIR).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
AGENTS_DIR = BASE_DIR / "AGENTS"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "00_COORDINATOR.md": """# AGENTS/00_COORDINATOR.md — Instruções para o Coordenador

Você é o responsável por: Iniciar o fluxo, distribuir tarefas, garantir que cada agente tem as informações que precisa, coordenar dependências e validar entregáveis.

--- 
(SUMMARY)
- Fases: Teoria única, Tutoriais, Testes, Integração.
- Agentes: EM Specialist, Senior Dev, Dev-1, Dev-2, Dev-3.
- Checkpoints e timeline já definidos (2 semanas).
- See PLANEJAMENTO.md for details.

Checklist (Dia 1):
- Crie /AGENTS/
- Copie os ficheiros de agente
- Execute Sphinx setup (instalar dependências)
- Distribua arquivos e confirme leitura

Dia-a-dia, dependências, checkpoints, comunicação e templates de mensagens estão aqui para orientar o sprint.

(Use este ficheiro como referência principal do coordenador.)
""",
    "01_EM_SPECIALIST.md": """# AGENTS/01_EM_SPECIALIST.md — Instruções para o Especialista EM

Você é: autoridade em física. Valida fórmulas, define tolerâncias, revisa tutoriais e assina o sign-off final.

Principais tarefas:
- Auditoria de conftest.py: fc_te10, propagation_constant, te_impedance.
- Revisão do core FDTD: update_H, update_E, CFL.
- Revisão de tutoriais (PEC, cavidade).
- Produzir relatório de auditoria (AUDITORIA DE FÍSICA).
- Assinar FINAL_VALIDATION_SIGN_OFF.md após validação.

Referências: Taflove, Pozar, Griffiths.
Tolerâncias sugeridas: fc ±0.5%, impedância ±3%, Γ ±2%, cavidade ±2%.

Comandos de ativação:
- @EM_Specialist Revisa a física: conftest.py ...
- @EM_Specialist Valida tutorial: Tutorial/04_boundaries/pec_reflection.py
- @EM_Specialist Sign-Off final

Use o template de auditoria fornecido e documente cada decisão.
""",
    "02_SENIOR_DEV.md": """# AGENTS/02_SENIOR_DEV.md — Instruções para o Revisor Sénior

Você é: autoridade em qualidade de código. Garante convenções, testes, CI e revisão final.

Checklist de review (12 pontos):
1. Naming consistente (snake_case, PascalCase...
2. Testes: cobertura e casos limites.
3. Constantes centralizadas (emsim.constants).
4. Fronteiras (f=fc, f<<fc, f>>fc).
5. Execução from project root.
6. Docstrings e type hints.
7. Linters (flake8/pylint).
8. Performance (sem loops Python nos kernels).
9. Tests rodáveis e marcados (slow/validation).
10. Dependências em pyproject.toml.
11. Backwards compatibility.
12. Git hygiene (mensagens, branch).

Tarefas principais:
- Criar docs/CODING_STANDARDS.md
- Revisar PRs Dev-1/2/3 com checklist
- Preparar FINAL_REVIEW_CHECKLIST.md
- Coordenar merges após EM Specialist approve

Comandos:
- @Senior_Dev Code review: PR #X
- @Senior_Dev Define padrões
- @Senior_Dev Sign-off final
""",
    "03_DEV_1.md": """# AGENTS/03_DEV_1.md — Instruções para Dev-1

Você é: corrige fórmulas analíticas e conftest.

Tarefas imediatas (Fase 1):
- Corrigir tests/conftest.py:
  - fc_te10_wr42() → C0 / (2 * a) usando fixture wr42_dimensions
  - propagation_constant(f, fc): usar f no numerador para evanescente
  - te_impedance(f, fc): retornar j*ETA0/sqrt((fc/f)**2 - 1) para f < fc; handle f==fc

- Adicionar testes unitários em tests/unit/test_conftest.py cobrindo f>fc, f<fc, f==fc.

Implementação mínima: não refatore além do necessário. Prepare PR com descrição clara.

Comando de ativação (coordenador):
@Dev_1 Implementa — Correções conftest.py

PR template e testes exemplares estão no AGENTS docs.
""",
    "04_DEV_2.md": """# AGENTS/04_DEV_2.md — Instruções para Dev-2

Você é: implementa tutoriais e comparações teoria vs simulação.

Tarefas:
- Tutorial PEC: Tutorial/04_boundaries/pec_reflection.py
  - Medir Γ (opção amplitude ratio ou SWR)
  - Comparar Γ_sim vs Γ_theory (-1) com tolerância ±2%
  - Produzir plots e salvar PNG

- Cavidade/Guias: Tutorial/06_waveguides_modes/README.md
  - Documentar que Ez excita TM apenas
  - Plotar linhas teóricas (TE+TM) e dados simulação (TM)
  - Tolerância ±2% para frequências de pico

Opcional: Tutorial dielétrico (medir velocidade em material).

Comandos:
- @Dev_2 Implementa: PEC tutorial
- @Dev_2 Documenta: cavity README

Submeter PR para EM Specialist (física) e Senior Dev (código).
""",
    "05_DEV_3.md": """# AGENTS/05_DEV_3.md — Instruções para Dev-3

Você é: implementa testes de validação, unifica dimensões (WR42), documenta validação.

Tarefas principais:
- Criar/atualizar tests/validation/:
  - test_te10_cutoff.py (cutoff e phase velocity)
  - test_pec_reflection.py (Γ measured)
  - test_cavity_resonance.py (TM110)
  - test_te_impedance.py (above/below cutoff)
  - test_wr42_dimensions_consistent.py (conftest vs Simulations/WR42/config.yaml)

- Unificar WR42 canonical dimensions in conftest fixture and Simulations/WR42/config.yaml:
  a = 10.7e-3, b = 4.3e-3

- Criar docs/validation.md com matrix teoria–simulação e instruções para rodar validação.

Comandos:
- @Dev_3 Implementa: validation tests
- @Dev_3 Unifica: WR42 dims
- @Dev_3 Documenta: docs/validation.md

PR template and expected pytest commands included in AGENTS documentation.
""",
}

for name, content in files.items():
    path = AGENTS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")

print("Done. Files created in:", AGENTS_DIR)