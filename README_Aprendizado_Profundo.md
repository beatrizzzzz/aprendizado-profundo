# Aprendizado Profundo — Implementação de Redes Neurais com Python e PyTorch

Repositório desenvolvido para a disciplina de **Aprendizado Profundo**, do curso de Ciência de Dados e Inteligência Artificial da Universidade Federal da Paraíba (UFPB), período 2026.2.

O projeto reúne cinco exercícios práticos envolvendo implementação, treinamento e avaliação de redes neurais artificiais. Foram exploradas arquiteturas MLP (*Multilayer Perceptron*), CNN (*Convolutional Neural Network*) e autoencoders, aplicadas a problemas de regressão, classificação, extração de características e reconstrução de imagens.

Os códigos incluem preparação dos dados, definição das arquiteturas, treinamento, validação, visualização de resultados e análise quantitativa do desempenho.

## 1. Aproximação de funções e regressão com MLP

### Questão 1(a) — Aproximação de função matemática

Foi implementada uma rede neural MLP para aproximar a função:

$$
f(x)=3\sin(2x)+2\cos(5x)+0{,}5x^2
$$

no intervalo $0 \leq x \leq 10$.

**Arquitetura utilizada:**

- Entrada: 1 variável numérica.
- Primeira camada oculta: 64 neurônios, com ativação Tanh.
- Segunda camada oculta: 64 neurônios, com ativação Tanh.
- Saída: 1 neurônio para regressão.

**Estrutura:** `1 → 64 → 64 → 1`

O código gera pares de entrada e saída a partir da função matemática, separa os dados em conjuntos de treinamento, validação e teste e ajusta os pesos da rede para minimizar o erro entre os valores previstos e os valores reais.

A avaliação inclui curvas de perda, comparação visual entre a função original e a função aproximada e métricas MAE, MSE e RMSE.

### Questão 1(b) — Regressão com Diabetes Dataset

Foi desenvolvida uma MLP para prever uma medida quantitativa de progressão do diabetes a partir de dez atributos clínicos disponíveis no conjunto de dados do Scikit-learn.

A implementação contempla análise exploratória, preparação dos atributos, treinamento supervisionado e avaliação das previsões.

As métricas utilizadas incluem MAE, MSE, RMSE e coeficiente de determinação R², além de gráficos comparando os valores reais e preditos.

## 2. Classificação multiclasse com MLP — Wine Quality

O segundo exercício utiliza o Wine Quality Dataset, disponibilizado pela UCI, para classificar vinhos em três categorias:

- Baixa qualidade: notas de 0 a 4.
- Média qualidade: notas 5 e 6.
- Alta qualidade: notas de 7 a 10.

Foram combinadas informações de vinhos tintos e brancos, utilizando características físico-químicas e um indicador do tipo de vinho.

### Arquitetura da rede

`12 → 64 → 32 → 3`

- Camada de entrada: 12 atributos.
- Primeira camada oculta: 64 neurônios com ReLU.
- Dropout: 25%.
- Segunda camada oculta: 32 neurônios com ReLU.
- Dropout: 15%.
- Camada de saída: 3 neurônios, correspondentes às classes.
- Total: 3.011 parâmetros treináveis.

### Estratégia de treinamento

O código utiliza Cross-Entropy ponderada para reduzir o impacto do desbalanceamento entre as classes, atribuindo pesos diferentes aos erros de classificação.

O treinamento emprega validação e parada antecipada (*early stopping*), com seleção do modelo de acordo com o desempenho de validação.

A avaliação é realizada por meio de matriz de confusão, acurácia, precisão, recall e F1-score, incluindo análise individual das três classes.

## 3. Classificação de imagens — Comparação entre MLP e CNN

Nesta etapa, foram implementados dois modelos independentes para classificar imagens do CIFAR-10, composto por dez categorias.

As imagens RGB possuem resolução de 32 × 32 pixels, totalizando 3.072 valores por imagem.

O conjunto foi dividido em:

- Treinamento: 45.000 imagens.
- Validação: 5.000 imagens.
- Teste: 10.000 imagens.

Os pixels são normalizados para o intervalo [-1, 1], utilizando a transformação `pixel / 127.5 - 1`.

### Arquitetura MLP

```text
Entrada RGB (3 × 32 × 32)
         ↓
Flatten (3072)
         ↓
Linear (3072 → 256)
         ↓
ReLU + Dropout (0.30)
         ↓
Linear (256 → 128)
         ↓
ReLU + Dropout (0.30)
         ↓
Linear (128 → 10)
         ↓
Logits das 10 classes
```

A MLP transforma a imagem em um vetor unidimensional antes do processamento. Dessa forma, utiliza camadas totalmente conectadas para aprender relações entre os valores dos pixels, sem preservar explicitamente sua organização espacial.

**Total: 820.874 parâmetros treináveis.**

### Arquitetura CNN

```text
Entrada RGB (3 × 32 × 32)
         ↓
Conv2d (3 → 16, kernel 3×3, padding 1)
ReLU + MaxPool2d (2)
         ↓
16 × 16 × 16
         ↓
Conv2d (16 → 32, kernel 3×3, padding 1)
ReLU + MaxPool2d (2)
         ↓
32 × 8 × 8
         ↓
Conv2d (32 → 64, kernel 3×3, padding 1)
ReLU + MaxPool2d (2)
         ↓
64 × 4 × 4
         ↓
Flatten (1024)
         ↓
Linear (1024 → 128)
ReLU + Dropout (0.30)
         ↓
Linear (128 → 10)
         ↓
Logits das 10 classes
```

A CNN utiliza três blocos convolucionais para extrair características espaciais das imagens. Cada bloco combina convolução, ativação ReLU e MaxPooling, permitindo aprender representações progressivamente mais complexas.

**Total: 156.074 parâmetros treináveis.**

### Configuração do treinamento

Ambos os modelos foram treinados com:

- Framework: PyTorch.
- Função de perda: Cross-Entropy.
- Otimizador: Adam.
- Taxa de aprendizado inicial: 0,001.
- Weight decay: 0,0001.
- Batch size: 256.
- Máximo de 30 épocas.
- Early stopping com paciência de 7 épocas.
- Scheduler ReduceLROnPlateau.
- Semente aleatória: 42.

A seleção dos pesos considera a menor perda de validação, mantendo o conjunto de teste reservado para a avaliação final.

### Resultados

| Modelo | Acurácia no teste |
|---|---:|
| MLP | 56,13% |
| CNN | 72,93% |

A CNN apresentou desempenho superior, com diferença de 16,80 pontos percentuais de acurácia, utilizando menos parâmetros treináveis.

Os resultados são analisados por curvas de treinamento e validação, matrizes de confusão e comparação das classificações.

## 4. Visualização de filtros e mapas de ativação da CNN

A quarta questão utiliza a CNN desenvolvida na Questão 3 para investigar as representações internas aprendidas durante o treinamento.

Não foi construída uma nova arquitetura: a análise utiliza as camadas convolucionais do modelo já implementado.

### Visualização dos filtros

Foram extraídos os pesos da primeira camada convolucional:

`Conv2d(3, 16, kernel_size=3, padding=1)`

Essa camada contém 16 filtros, cada um com dimensões 3 × 3 × 3, considerando os três canais RGB.

A visualização permite acompanhar a evolução dos filtros durante o treinamento e observar padrões associados a cores, contrastes e estruturas locais.

### Mapas de ativação

Foram analisadas ativações intermediárias da segunda camada convolucional:

`Conv2d(16, 32, kernel_size=3, padding=1)`

Essa camada produz 32 canais de características com resolução espacial de 16 × 16, antes da segunda operação de pooling.

Os mapas permitem observar como diferentes filtros respondem às regiões das imagens de entrada.

### Análise das classificações

Também foram selecionados exemplos do conjunto de teste para visualizar:

- Imagem original.
- Classe verdadeira.
- Classe prevista pela CNN.
- Probabilidade associada à predição.
- Exemplos de classificações corretas e incorretas.

Essa etapa complementa a avaliação quantitativa ao investigar o comportamento interno da rede convolucional.

## 5. Autoencoders e remoção de ruído — Fashion-MNIST

A quinta questão explora aprendizado de representações comprimidas e reconstrução de imagens utilizando o Fashion-MNIST.

Cada imagem possui resolução de 28 × 28 pixels em escala de cinza, correspondendo a 784 valores.

O conjunto foi dividido em 54.000 imagens de treinamento, 6.000 de validação e 10.000 de teste.

### Arquitetura dos autoencoders

Foram implementados três autoencoders totalmente conectados, diferenciados exclusivamente pela dimensão do espaço latente: 100, 50 e 30.

**Encoder:**

```text
Imagem (1 × 28 × 28)
         ↓
Flatten (784)
         ↓
Linear (784 → 256) + ReLU
         ↓
Linear (256 → 128) + ReLU
         ↓
Linear (128 → z)
         ↓
Representação latente
```

**Decoder:**

```text
Representação latente (z)
         ↓
Linear (z → 128) + ReLU
         ↓
Linear (128 → 256) + ReLU
         ↓
Linear (256 → 784) + Sigmoid
         ↓
Unflatten (1 × 28 × 28)
         ↓
Imagem reconstruída
```

O encoder comprime a imagem em uma representação de menor dimensionalidade, enquanto o decoder tenta reconstruir a imagem original a partir desse vetor.

A função Sigmoid na saída restringe os pixels reconstruídos ao intervalo [0, 1].

### Treinamento e avaliação

Os modelos foram treinados com:

- Função de perda: MSE.
- Otimizador: Adam.
- Taxa de aprendizado inicial: 0,001.
- Batch size: 256.
- Máximo de 30 épocas.
- Early stopping com paciência de 5 épocas.
- Scheduler ReduceLROnPlateau.

O desempenho foi avaliado por MSE e PSNR, além da comparação visual entre imagens originais e reconstruídas.

| Espaço latente | MSE | PSNR |
|---|---:|---:|
| 100 | 0,010250 | 20,88 dB |
| 50 | 0,010393 | 20,82 dB |
| 30 | 0,010387 | 20,82 dB |

### Denoising autoencoder

Foi implementado um autoencoder para reconstruir imagens limpas a partir de versões corrompidas por ruído.

O modelo utiliza a mesma estrutura encoder-decoder, com espaço latente de 50 dimensões.

Durante o treinamento, é adicionado ruído gaussiano com média zero e desvio-padrão de 0,30 às imagens de entrada. Os valores são limitados ao intervalo [0, 1].

A imagem original, sem ruído, é utilizada como alvo da função de perda. Assim, a rede aprende a reconstruir a informação visual relevante, em vez de reproduzir o ruído recebido.

### Resultados da remoção de ruído

| Condição | MSE | PSNR |
|---|---:|---:|
| Imagem com ruído | 0,052677 | 12,80 dB |
| Autoencoder convencional | 0,028232 | 15,69 dB |
| Denoising autoencoder | 0,013529 | 19,39 dB |

O denoising autoencoder apresentou menor erro de reconstrução e maior PSNR, demonstrando a vantagem do treinamento com entradas ruidosas e alvos limpos nesse experimento.

## Tecnologias utilizadas

- **Python:** linguagem principal de implementação.
- **PyTorch:** construção das arquiteturas, cálculo de gradientes e treinamento.
- **NumPy:** manipulação de arrays e operações numéricas.
- **Pandas:** organização e análise de dados tabulares.
- **Scikit-learn:** pré-processamento, divisão dos dados e métricas.
- **Matplotlib:** visualização de curvas, matrizes de confusão, filtros e reconstruções.
- **Jupyter Notebook:** organização e execução dos experimentos.

## Considerações finais

Os experimentos permitem comparar diferentes arquiteturas neurais e compreender como suas características influenciam o aprendizado.

A comparação entre MLP e CNN evidencia a importância das convoluções para o processamento de imagens. A visualização dos filtros e mapas de ativação permite investigar as representações aprendidas, enquanto os autoencoders demonstram aplicações de compressão de características e reconstrução de dados.

O repositório documenta não apenas os resultados, mas também as decisões de implementação, arquiteturas e procedimentos de avaliação utilizados nos experimentos.

---

**Universidade Federal da Paraíba — UFPB**  
**Curso:** Ciência de Dados e Inteligência Artificial  
**Disciplina:** Aprendizado Profundo  
**Período:** 2026.2
