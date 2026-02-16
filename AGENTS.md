# Fluxo do agente — EMSIM

Este projeto usa um **fluxo em três modos**. O utilizador deve pedir explicitamente cada modo.

- **Implementação (5 passos):** "Implementa X", "Desenvolve a feature Y" — o agente estrutura em 5 passos lógicos (grid, materiais, campos, fronteiras/fontes, testes ou pós-processamento).
- **Revisão de código:** "Revisa este código como revisor sénior", "Code review" — o agente aplica o checklist de revisão (naming, testes, constantes, fronteiras, execução na raiz).
- **Revisão de física:** "Revisa a física", "Confirma teoria vs simulação", "Validar resultados EM" — o agente verifica fórmulas, unidades e coerência com a teoria (Tutorial/common/theory.py, emsim).

As definições detalhadas e listas de verificação estão em `.cursor/rules/` (workflow-roles, senior-review, em-simulation-expert).
