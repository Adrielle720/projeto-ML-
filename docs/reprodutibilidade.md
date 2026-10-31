# 11. Reprodutibilidade

| Item | Valor |
|---|---|
| Semente aleatória | `random_state=42` em todos os splits, buscas e modelos |
| Divisão | 80/20 estratificada → 4.088 treino / 1.022 teste |
| Validação | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Métrica de seleção | `average_precision` |
| Limiar final | 0,08 (maior F2 fora da dobra no treino) |
| Melhor modelo | `RandomForestClassifier(n_estimators=200, max_depth=4, max_features="sqrt", min_samples_leaf=1)` com o pipeline modificado |

**Como rodar:** abrir `stroke_prediction.ipynb` (Colab ou Jupyter) e executar **todas as células em ordem**. A primeira célula de código baixa o dataset com `kagglehub`, e as demais dependem do `dataset` carregado nela. A busca da Random Forest (150 ajustes, repetida na seção 7.6) leva alguns minutos. O `shap` é instalado no próprio notebook (`%pip install shap`).

**Avisos esperados:** o scikit-learn 1.8 emite `FutureWarning` sobre o parâmetro `penalty` da `LogisticRegression`, que será substituído por `l1_ratio` em versões futuras. O aviso não afeta os resultados.
