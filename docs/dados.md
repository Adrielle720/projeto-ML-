# 1. Os dados

## 1.1 Origem

O dataset foi publicado no Kaggle por fedesoriano e é descrito apenas como "confidencial". **Não há informação sobre o hospital ou país de origem, o período de coleta, o critério de inclusão dos pacientes ou a forma de diagnóstico do AVC.** Essa falta de documentação é uma das limitações discutidas na seção 10.

O download é feito no próprio notebook com `kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")`, sem precisar de login, já que o dataset é público.

## 1.2 Dimensões e variáveis

São **5.110 pacientes** (linhas) e **12 colunas**.

| Variável | Tipo | Descrição | Valores |
|---|---|---|---|
| `id` | inteiro | Identificador do paciente | 5.110 únicos (removido da modelagem) |
| `gender` | categórica | Sexo | Female 2.994 · Male 2.115 · Other 1 |
| `age` | contínua | Idade (anos) | 0,08 a 82; média 43,2; mediana 45 |
| `hypertension` | binária | Hipertensão | 1 = 498 (9,75%) |
| `heart_disease` | binária | Doença cardíaca | 1 = 276 (5,40%) |
| `ever_married` | categórica | Já foi casado(a) | Yes 3.353 · No 1.757 |
| `work_type` | categórica | Tipo de trabalho | Private 2.925 · Self-employed 819 · children 687 · Govt_job 657 · Never_worked 22 |
| `Residence_type` | categórica | Moradia | Urban 2.596 · Rural 2.514 |
| `avg_glucose_level` | contínua | Glicose média (mg/dL) | 55 a 272; média 106,2; mediana 91,9; dp 45,3 |
| `bmi` | contínua | Índice de massa corporal | 201 ausentes (3,93%); média 28,9; mediana 28,1; dp 7,85 |
| `smoking_status` | categórica | Tabagismo | never smoked 1.892 · Unknown 1.544 · formerly smoked 885 · smokes 789 |
| **`stroke`** | **binária (alvo)** | **Teve AVC** | **1 = 249 (4,87%) · 0 = 4.861 (95,13%)** |

## 1.3 Qualidade dos dados

| Verificação | Resultado | Decisão |
|---|---|---|
| Valores nulos (`NaN`) | Só no `bmi`: 201 (3,93%) | Preencher com a mediana (APS1). Na APS2, também criar um indicador de ausência (seção 7.5) |
| Valores `None` em colunas de texto | Nenhum | — |
| Zeros impossíveis (`bmi` = 0, glicose = 0) | Nenhum. Os zeros só aparecem nas binárias, onde são legítimos | — |
| `age` fora de [0, 121] | Nenhum | Mantido. Idades fracionárias (0,08) correspondem a bebês |
| `avg_glucose_level` fora de [30, 600] | Nenhum | — |
| `bmi` fora de [10, 80] | 2 pacientes: 97,6 e 92,0 (17 e 38 anos, sem AVC) | Mantidos. São valores extremos mas fisicamente possíveis, e o `StandardScaler` e as árvores toleram esses pontos |
| `gender = "Other"` | 1 paciente | Substituído pela moda (`Female`). Uma categoria com um único exemplo não permite aprender nada e geraria uma coluna quase constante no one-hot |
| `smoking_status = "Unknown"` | 1.544 (30,22%) | Mantido como categoria própria. É um volume grande demais para descartar, e a própria ausência é informativa: a taxa de AVC no `Unknown` é de 3,0%, a menor de todas, porque muitas crianças não têm esse dado |

## 1.4 O desbalanceamento

Só **249 pacientes (4,87%)** tiveram AVC. Essa é a característica que mais influencia o trabalho:
- Um modelo que sempre responde "sem AVC" acerta **95,1%**. Por isso a **acurácia não serve como métrica principal** (seção 6).
- Com 80/20, sobram só **199 AVCs no treino** e **50 no teste**. Cada AVC a mais ou a menos detectado no teste muda o recall em 2 pontos percentuais.
