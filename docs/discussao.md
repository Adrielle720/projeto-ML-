# 10. APS2 — Discussão crítica

## 10.1 Limitações do melhor modelo

**1. Associação não é causa, e as explicações herdam essa limitação.**
- O SHAP e as importâncias mostram **como o modelo usa** cada variável, não o efeito causal dela sobre o AVC.
- `ever_married` aparece com efeito positivo só por estar correlacionada com a idade. O mesmo vale para os coeficientes de `children` e `Never_worked` (seção 9.1).
- Quando variáveis são correlacionadas, o crédito é dividido entre elas de forma que depende do algoritmo. Uma explicação SHAP é uma **estimativa a ser verificada**, não uma justificativa clínica (Lundberg & Lee, 2017; Molnar, 2022).
- Usar o modelo para recomendar intervenções ("reduzir X diminui o risco") seria um erro: para isso seria preciso um desenho causal, não um modelo preditivo.

**2. Poucos eventos e estimativas instáveis.**
- O treino tem **199 AVCs** e o teste, **50**. Em modelos de predição clínica, o número de eventos por parâmetro determina a estabilidade do modelo (Peduzzi et al., 1996; Riley et al., 2020).
- A instabilidade aparece em todo o trabalho:
  - desvio de ±0,04 a ±0,06 na AP entre dobras, maior que várias das diferenças entre modelos;
  - coeficientes que divergem em categorias sem eventos;
  - no teste, cada AVC detectado vale 2 pontos de recall.
- A vantagem da Random Forest sobre a logística (40 × 39 AVCs detectados no teste) **não é estatisticamente distinguível**. Os números deste relatório não devem ser lidos como precisos na segunda casa decimal.

## 10.2 Limitações dos dados

**1. Origem desconhecida, sem documentação, e a possibilidade de vazamento no `bmi` ausente.**
- O dataset não informa a instituição de origem, o país, o período, o critério de inclusão nem como o AVC foi diagnosticado.
- Sem uma documentação no formato de *datasheet* (Gebru et al., 2021), não dá para saber **a quem o modelo se aplica**, se há viés de seleção ou se os pacientes com e sem AVC foram registrados da mesma forma.
- O caso do `bmi` ausente mostra o risco concreto disso. A taxa de AVC é 5 vezes maior quando o `bmi` falta (seção 7.6), o que sugere que a forma de coleta foi diferente para os casos. Se for isso, o melhor modelo explora um artefato de registro.
- Também não há uma segunda base para **validação externa**, sem a qual o desempenho em outra população é desconhecido.

**2. Dados transversais e fatores de risco ausentes.**
- Não há informação temporal: não se sabe se a hipertensão, a glicose ou o tabagismo foram medidos **antes ou depois** do AVC. Isso permite **causalidade reversa**. Por exemplo, a taxa maior entre ex-fumantes (7,8%) do que entre fumantes (5,4%) pode refletir pacientes que pararam de fumar **depois** do evento.
- Faltam fatores de risco estabelecidos para AVC, como **fibrilação atrial, pressão arterial medida (e não só o diagnóstico de hipertensão), colesterol, uso de anti-hipertensivos, diabetes diagnosticado e histórico familiar**. Vários deles fazem parte de escores clínicos consolidados, como o perfil de risco de Framingham (Wolf et al., 1991).
- `smoking_status = "Unknown"` cobre **30%** dos pacientes.
- O falso negativo da seção 9.3 (14 anos, sem fatores) mostra o efeito prático: AVCs com causas fora dessas variáveis são invisíveis para qualquer modelo treinado com elas.

## 10.3 Ações para operacionalização e manutenção

**1. Monitoramento contínuo e recalibração periódica.**
- Em produção, a população atendida muda com o tempo: perfil etário, prevalência de AVC, equipamentos e protocolos de medição. Isso degrada o modelo silenciosamente, sem nenhum erro aparente no sistema (Sculley et al., 2015).
- O que monitorar:
  - **a distribuição das variáveis de entrada**, por exemplo a fração de `bmi` ausente, que é crítica dada a seção 7.6;
  - **a distribuição das probabilidades previstas**;
  - **AP, recall e calibração**, assim que os desfechos reais forem conhecidos.
- A calibração é o ponto que mais se degrada quando a prevalência muda (Van Calster et al., 2019). O modelo deve ser recalibrado ou retreinado quando o monitoramento indicar desvio, e o limiar reavaliado junto.

**2. Uso como apoio à decisão, com documentação e governança.**
- O modelo deve entrar no fluxo clínico como **ferramenta de triagem** para acompanhamento preventivo, com decisão final do profissional de saúde, **nunca como diagnóstico automático**.
- O limiar (hoje 0,08) deve ser definido com a equipe clínica, conforme a capacidade de absorver alarmes falsos. Com o limiar atual, seriam ~18% dos pacientes sem AVC sinalizados.
- O modelo deve ser publicado com um ***model card*** (Mitchell et al., 2019) que registre:
  - os dados de treino e suas limitações (seção 10.2);
  - as métricas **por subgrupo** (sexo, faixa etária), para detectar desempenho desigual;
  - o uso pretendido e o não pretendido.
- Por tratar de dados de saúde, que são dados pessoais sensíveis, o uso precisa respeitar a **LGPD** (Lei 13.709/2018), incluindo o direito do titular de pedir revisão de decisões tomadas só com base em tratamento automatizado (art. 20). Isso reforça a necessidade de um humano na decisão final.
