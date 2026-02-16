# Planejamento: robustez e consistência teoria vs simulação

Este documento consolida o plano para tornar o simulador EMSIM robusto e consistente com a teoria electromagnética, com divisão de tarefas por papel (3 devs, 1 sénior, 1 especialista EM) e lista concreta de alterações.

---

## 1. Objetivos

- **Uma única fonte** de fórmulas analíticas (conftest e tutoriais alinhados).
- **Comparações explícitas** nos tutoriais: valor teórico e valor medido em cada exemplo.
- **Testes de validação** com tolerâncias claras e mensagens de falha úteis.
- **Dimensões WR42** coerentes em todo o projeto (config, testes, tutoriais).
- **Documentação** que explique o que se valida e onde está a teoria de referência.

---

## 2. Inconsistências identificadas (a corrigir)

| # | Problema | Onde | Ação |
|---|----------|------|------|
| 2.1 | `fc_te10` na fixture `wr42_dimensions` não corresponde a `a=10.7e-3` (14.05 vs ~14.01 GHz) | [tests/conftest.py](tests/conftest.py) | Corrigir `fc_te10` para `C0/(2*a)` com o `a` da fixture; documentar. |
| 2.2 | `propagation_constant` evanescente usa `fc` em vez de `f` na expressão de α | [tests/conftest.py](tests/conftest.py) | Usar `f` na fórmula: `1j * 2*pi*f*sqrt((fc/f)**2-1)/C0`. |
| 2.3 | `te_impedance` para f ≤ fc devolve 0; deveria ser impedância imaginária | [tests/conftest.py](tests/conftest.py) | Devolver `1j*ETA0/sqrt((fc/f)**2-1)` ou delegar em `emsim.modes.rectangular`. |
| 2.4 | Script PEC não calcula Γ a partir da simulação; só imprime teoria | [Tutorial/04_boundaries/pec_reflection.py](Tutorial/04_boundaries/pec_reflection.py) | Estimar Γ (incidente/reflectido ou onda estacionária) e imprimir Γ_sim vs Γ_teoria. |
| 2.5 | Teste PEC não verifica Γ medido ≈ -1 | [tests/validation/test_pec_reflection.py](tests/validation/test_pec_reflection.py) | Opcional: calcular Γ nos testes e assert \|Γ+1\| < tolerância. |
| 2.6 | Cavidade: gravação de Ez excita sobretudo modos TM; linhas teóricas incluem TE e TM | Tutorial 06 README / notebook / script | Documentar: "Ez → modos TM; linhas teóricas incluem TE e TM". |
| 2.7 | Fórmulas duplicadas em conftest e Tutorial/common/theory.py | Vários | Unificar: conftest importar de um módulo analítico único (ou documentar alinhamento). |

---

## 3. Divisão por papéis e fases

### Papéis

| Papel | Responsabilidade |
|-------|------------------|
| **Dev 1** | Fórmulas analíticas, conftest, módulo de teoria única. |
| **Dev 2** | Tutoriais e scripts "teoria vs simulação" (PEC Γ, dielétrico, cavidade, S-params). |
| **Dev 3** | Testes de validação, WR42, tolerâncias, CI. |
| **Sénior** | Code review, convenções, critérios de "done", documentação de processo. |
| **Especialista EM** | Validação das fórmulas, método de medição de Γ, revisão física dos tutoriais. |

### Fase 1 — Teoria única e conftest (Dev 1 + Especialista EM)

- **Dev 1**
  - Corrigir em [tests/conftest.py](tests/conftest.py):
    - `wr42_dimensions`: `fc_te10 = C0 / (2 * 10.7e-3)` (ou manter `a` e documentar).
    - `propagation_constant` (f < fc): usar `f` em vez de `fc`.
    - `te_impedance` (f ≤ fc): devolver impedância imaginária.
  - Opcional: criar módulo único (ex. [Tutorial/common/theory.py](Tutorial/common/theory.py) ou `emsim.analytics`) e fazer conftest importar as funções em vez de reimplementar.
- **Especialista EM**
  - Validar fórmulas (corte, cavidade, β, Z_TE, Γ) antes e depois das alterações.
  - Aprovar especificação das funções analíticas.

**Entregável:** Conftest coerente; fórmulas num único sítio (ou documentado).

---

### Fase 2 — Tutoriais e comparação teoria/sim (Dev 2 + Especialista EM)

- **Dev 2**
  - [Tutorial/04_boundaries/pec_reflection.py](Tutorial/04_boundaries/pec_reflection.py): calcular Γ a partir dos sinais; imprimir Γ_sim e Γ_teoria.
  - [Tutorial/05_materials](Tutorial/05_materials): garantir uso do mesmo `measure_wave_speed`/teoria.
  - [Tutorial/06_waveguides_modes](Tutorial/06_waveguides_modes): no README e no notebook/script, documentar Ez → modos TM; linhas teóricas TE+TM.
  - [Tutorial/08_postprocessing](Tutorial/08_postprocessing): fc TE10 com mesmas dimensões que a simulação WR42.
- **Especialista EM**
  - Definir método recomendado para medir Γ (incidente/reflectido ou onda estacionária).
  - Revisar texto e figuras dos tutoriais; validar um run de cada script.

**Entregável:** Scripts a mostrar valor teórico e valor medido; documentação cavidade.

---

### Fase 3 — Testes e WR42 (Dev 3 + Sénior)

- **Dev 3**
  - Garantir que testes que usam `analytical_solutions` passam após Fase 1.
  - Opcional: teste que usa `propagation_constant` para f < fc (travar regressão).
  - [tests/validation/test_pec_reflection.py](tests/validation/test_pec_reflection.py): opcional verificação Γ medido ≈ -1.
  - Unificar dimensões WR42: uma convenção (10.7/4.3 ou 10.67/4.32) em conftest, [Simulations/WR42/config.yaml](Simulations/WR42/config.yaml), tutoriais e testes; documentar.
  - Usar tolerâncias centralizadas (fixture `tolerances` no conftest) nos testes de validação.
- **Sénior**
  - Code review dos testes; definir/ajustar tolerâncias padrão.
  - Garantir que a suíte de validação corre no CI (markers `validation`, `slow`).

**Entregável:** Testes alinhados com a teoria; WR42 consistente; tolerâncias definidas.

---

### Fase 4 — Integração e critérios (Sénior + Especialista EM)

- **Sénior**
  - Checklist de aceitação (fórmulas únicas, tutoriais com teoria+sim, testes com tolerâncias, WR42 documentado).
  - Documento curto (ex. [docs/validation.md](docs/validation.md)): o que se valida, como correr testes de validação, onde está a teoria.
- **Especialista EM**
  - Matriz teoria–simulação: tabela (velocidade, Γ PEC, fc TE10, f_mnp, v_diel/v_vac, Z_TE) com "fórmula" e "onde se valida".
  - Revisão final: executar tutoriais e testes e confirmar resultados plausíveis.

**Entregável:** Critérios documentados; matriz teoria–simulação; validação final.

---

## 4. Ordem de execução sugerida

```
1. Especialista EM: especificação das fórmulas e método Γ
2. Dev 1: correções conftest (+ módulo único se opcional)
3. Especialista EM: aprovar fórmulas
4. Dev 2: PEC Γ, tutoriais, doc cavidade
5. Especialista EM: revisão tutoriais
6. Dev 3: testes, WR42, tolerâncias
7. Sénior: review testes e docs
8. Sénior + Especialista EM: critérios, matriz, validação final
```

---

## 5. Uso do Cursor (modo Auto e regras)

- **Um agente, vários papéis:** O modo Auto pode executar as tarefas em sequência; os "papéis" definem-se por **prompt** (ex.: "actua como revisor sénior e revê este código" ou "valida as fórmulas como especialista EM").
- **Regras em `.cursor/rules/`:** Podem definir os três modos (implementação em 5 passos, revisor sénior, perito EM) e frases de ativação; **não** é possível escolher modelo por regra (apenas `description`, `globs`, `alwaysApply`).
- **Modelo:** Escolhido no dropdown do chat; recomenda-se um modelo forte (ex. Claude Opus ou GPT-4o) para revisão e validação EM; o mesmo modelo pode seguir as regras de cada papel quando o utilizador o pedir.

---

## 6. Ficheiros principais envolvidos

| Área | Ficheiros |
|------|-----------|
| Fórmulas / conftest | [tests/conftest.py](tests/conftest.py), [Tutorial/common/theory.py](Tutorial/common/theory.py), [emsim/modes/rectangular.py](emsim/modes/rectangular.py), [emsim/constants.py](emsim/constants.py) |
| PEC | [Tutorial/04_boundaries/pec_reflection.py](Tutorial/04_boundaries/pec_reflection.py), [tests/validation/test_pec_reflection.py](tests/validation/test_pec_reflection.py) |
| Cavidade / guia | [Tutorial/06_waveguides_modes/README.md](Tutorial/06_waveguides_modes/README.md), [Tutorial/06_waveguides_modes/run_cutoff_and_cavity.py](Tutorial/06_waveguides_modes/run_cutoff_and_cavity.py), [Tutorial/06_waveguides_modes/cutoff_and_cavity.ipynb](Tutorial/06_waveguides_modes/cutoff_and_cavity.ipynb), [tests/validation/test_cavity_resonance.py](tests/validation/test_cavity_resonance.py) |
| WR42 | [tests/conftest.py](tests/conftest.py), [Simulations/WR42/config.yaml](Simulations/WR42/config.yaml), [Tutorial/08_postprocessing/plot_s_params_with_theory.py](Tutorial/08_postprocessing/plot_s_params_with_theory.py), testes de integração |
| Documentação | [docs/PLANEJAMENTO.md](docs/PLANEJAMENTO.md) (este ficheiro), futuro [docs/validation.md](docs/validation.md) |

---

## 7. Critérios de conclusão

- [ ] Todas as fórmulas analíticas vêm de um único módulo (ou conftest documentado e alinhado).
- [ ] Conftest: `fc_te10`, `propagation_constant` (evanescente), `te_impedance` (abaixo do corte) corrigidos.
- [ ] Script PEC calcula e imprime Γ_sim vs Γ_teoria.
- [ ] Tutorial 06 documenta Ez → TM; linhas teóricas TE+TM.
- [ ] Dimensões WR42 unificadas e documentadas.
- [ ] Testes de validação com tolerâncias explícitas; opcional assert Γ ≈ -1 no teste PEC.
- [ ] Documento de validação (o que se valida, como correr testes) e matriz teoria–simulação aprovada pelo especialista EM.
