# 3. Protocolo experimental da modelagem

## 3.1 Separação treino/teste

| Conjunto | Pacientes | AVCs | Prevalência |
|---|---|---|---|
| Treino | 4.088 | 199 | 4,87% |
| Teste | 1.022 | 50 | 4,89% |

A divisão é a mesma da APS1 (80/20, estratificada, `random_state=42`). **O teste fica guardado:** nenhuma escolha de modelo, hiperparâmetro ou limiar usa o teste. Se ele fosse usado para escolher, o número final estaria otimista, porque o modelo seria escolhido justamente por ter ido bem naquele conjunto.

## 3.2 Validação cruzada

Toda comparação usa `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre o treino:
- **5 dobras:** cada paciente é validado uma vez, e o desempenho é a média de 5 medidas, com o desvio entre as dobras indicando a incerteza.
- **Estratificada:** cada dobra tem ~40 AVCs. Sem estratificação, alguma dobra poderia ficar com poucos casos e a métrica oscilaria muito.
- **`shuffle=True`:** evita que alguma ordem do arquivo afete as dobras.
- **O mesmo objeto `cv` em tudo:** os modelos são avaliados nas **mesmas** dobras, o que permite comparações pareadas, dobra a dobra.

## 3.3 Pipeline (sem vazamento)

```
Pipeline
├── ColumnTransformer
│   ├── numéricas  → SimpleImputer(median) → StandardScaler
│   └── categóricas → OneHotEncoder(drop="first", handle_unknown="ignore")
└── modelo
```

- **Dentro do `Pipeline`, a mediana, a média, o desvio e as categorias são aprendidos só no treino de cada dobra** e depois aplicados à validação. Isso reproduz o uso real, em que o modelo nunca viu o paciente novo.
- **`drop="first"`:** remove uma categoria de referência por variável, para que as colunas do one-hot não sejam redundantes com o intercepto. Assim a matriz tem posto completo e os coeficientes ficam identificáveis. O resultado são **15 colunas** (5 numéricas + 10 de one-hot), contra 20 na APS1.
- **`handle_unknown="ignore"`:** se uma categoria rara, como `Never_worked`, não aparecer no treino de alguma dobra, a validação não quebra.
