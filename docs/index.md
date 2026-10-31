# Previsão de AVC com Machine Learning

- **Dataset:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, fedesoriano)
- **Notebook:** `stroke_prediction.ipynb` (APS1 + modelagem da APS2)
- **Ferramentas:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn e SHAP

## Resumo executivo

- **Problema:** prever se um paciente teve AVC (`stroke` = 1) a partir de 10 variáveis demográficas e clínicas. É uma **classificação binária muito desbalanceada**: só **4,87%** dos 5.110 pacientes tiveram AVC.
- **Métrica principal:** **average precision (AP)**. Ela foca na classe rara e não depende de limiar de decisão. Um modelo sem informação tem AP igual à prevalência, cerca de 0,049.
- **Modelos comparados:** Dummy, regressão logística regularizada e Random Forest, todos com o mesmo pipeline da APS1 e validação cruzada estratificada em 5 dobras.
- **Melhor modelo:** **Random Forest com o pipeline modificado**, que adiciona um indicador de `bmi` ausente e log em glicose e `bmi`. AP de validação: **0,263 ± 0,061**, contra 0,193 da logística e 0,049 do Dummy.
- **Resultado no teste** (1.022 pacientes, 50 AVCs, limiar 0,08 escolhido pelo F2):

  | Métrica | Valor |
  |---|---|
  | Recall | 0,80 (40 de 50 AVCs encontrados) |
  | Precision | 0,183 |
  | F1 | 0,297 |
  | Accuracy | 0,815 |
  | ROC-AUC | 0,844 |
  | Average precision | 0,230 |

- **O que mais pesa nas previsões:** idade, glicose média, hipertensão, ausência de `bmi` e doença cardíaca.
- **Principal ressalva:** o ganho da mudança no pipeline vem de um padrão de coleta. A taxa de AVC é de ~20% entre os pacientes sem `bmi` registrado e de ~4% entre os demais. Esse sinal pode não existir em um uso real do modelo (seção 7.5).

## Conteúdo

1. [Os dados](dados.md)
2. [APS1 — Análise exploratória, pré-processamento e clustering](aps1.md)
3. [Protocolo experimental da modelagem](protocolo.md)
4. [Regressão linear sobre o `stroke` (estudo exploratório)](regressao-linear.md)
5. [Regressão logística](regressao-logistica.md)
6. [Métricas de classificação](metricas.md)
7. [APS2 — Seleção e implementação dos modelos](selecao-modelos.md)
8. [APS2 — Avaliação do melhor modelo no teste](avaliacao-teste.md)
9. [APS2 — Interpretação e explicabilidade](interpretacao.md)
10. [APS2 — Discussão crítica](discussao.md)
11. [Reprodutibilidade](reprodutibilidade.md)
12. [Referências](referencias.md)
