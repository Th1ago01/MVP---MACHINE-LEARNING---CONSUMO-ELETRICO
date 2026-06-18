# MVP — Previsão de Consumo Horário de Energia Elétrica

Previsão do consumo horário de energia elétrica (kWh) de uma fábrica de água mineral, tratado como um problema de **séries temporais / regressão supervisionada**. O objetivo é apoiar decisões operacionais (planejamento de equipamentos, redução de custo em horário de ponta) antecipando o consumo a partir do histórico recente e de variáveis de calendário.

> MVP da disciplina de Machine Learning & Analytics.

## Problema

A operação consome energia de forma fortemente sazonal (turnos de produção, paradas noturnas e de fim de semana). Hoje o planejamento é reativo, o que gera custo desnecessário em horário de ponta e dificulta identificar consumo anômalo. Este projeto constrói e compara modelos para prever o consumo da próxima hora e discute as limitações da solução.

- **Tipo de tarefa:** séries temporais (forecasting) tratado como regressão supervisionada
- **Variável-alvo:** `consumo_kwh` (consumo ativo por hora, em kWh)
- **Métrica principal:** RMSE | **Métrica percentual:** WAPE (mais confiável que MAPE neste caso, por causa das horas de consumo próximo de zero)
- **Critério de sucesso:** superar o baseline ingênuo em ≥20% no RMSE

## Dados

- **Fonte:** exportação do sistema da CCEE (Mercado Livre de Energia), de uma fábrica real. Informações sensíveis foram removidas antes da publicação.
- **Período:** jan/2024 a dez/2025 (17.544 registros horários)
- **Granularidade:** horária
- **Arquivo:** `consumo_fabrica_privado.csv`

O dataset é carregado diretamente no notebook por **URL pública** (raw do GitHub), sem necessidade de login, token ou upload manual.

## Como executar

O notebook roda do início ao fim sem configuração adicional. A forma recomendada é o Google Colab:

1. Abra o notebook no Google Colab.
2. Menu **Ambiente de execução → Reiniciar e executar tudo**.

Todas as dependências (`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`) já vêm pré-instaladas no Colab. Para rodar localmente:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
jupyter notebook MVP_consumo_energia.ipynb
```

A seed global está fixada (`SEED = 160200`) para reprodutibilidade.

## Abordagem

1. **Análise exploratória** — sazonalidade diária (pico às 10h, vale às 20h) e semanal, distribuição bimodal (parada vs. produção), decomposição da série.
2. **Engenharia de features** — features de calendário, encodings cíclicos (seno/cosseno) para hora/dia/mês/estação, lags de 1h a 168h e médias móveis. Winsorizing do alvo com limites calculados apenas no treino.
3. **Divisão temporal** — holdout cronológico 80/20, sem embaralhamento, para evitar vazamento de dados.
4. **Modelagem** — baseline (média histórica por hora) + Ridge, Random Forest, Gradient Boosting e SARIMA.
5. **Otimização** — `RandomizedSearchCV` com `TimeSeriesSplit(5)` sobre o Random Forest.
6. **Avaliação** — métricas no teste, análise de resíduos, erro por hora do dia e importância das features.

## Resultados

Modelo final: **Random Forest Otimizado**, avaliado no conjunto de teste.

| Modelo | RMSE (kWh) | MAE (kWh) | R² | WAPE |
|---|---:|---:|---:|---:|
| **Random Forest Otimizado** | **112,47** | 51,22 | 0,931 | 16,3% |
| Random Forest (padrão) | 113,80 | 52,51 | 0,929 | 16,7% |
| Gradient Boosting | 119,82 | 62,38 | 0,921 | 19,8% |
| Ridge | 168,10 | 104,22 | 0,845 | 33,1% |
| Baseline (média por hora) | 316,78 | 235,36 | 0,451 | 74,8% |
| SARIMA(1,0,1)(1,0,1)[24] | 370,26 | 243,78 | 0,249 | 77,5% |

O modelo final reduz o RMSE em **~65%** sobre o baseline — bem acima do critério de 20%.

## Limitações

- **Horizonte de 1 hora:** o modelo usa o consumo real da hora anterior (lags realizados); previsões para horizontes maiores exigem estratégia recursiva ou direta, com erro acumulado.
- **Sem variáveis exógenas** (temperatura, volume de produção, feriados, paradas de manutenção).
- **SARIMA vs. ML:** a comparação não é direta — o SARIMA prevê todo o horizonte de uma vez, enquanto os modelos de ML preveem 1 hora à frente com lags reais.

## Próximos passos

- Incorporar feriados, temperatura e volume de produção como features exógenas.
- Implementar previsão multi-step (6h/12h/24h) e medir a degradação do erro por horizonte.
- Avaliar modelos sequenciais (LSTM, Temporal Fusion Transformer) com mais histórico.

## Estrutura do repositório

```
.
├── MVP_consumo_energia.ipynb     # notebook principal (relatório executável)
├── consumo_fabrica_privado.csv   # dataset (dados horários, sem informações sensíveis)
└── README.md
```
