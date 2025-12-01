# Clusterização de Clientes por Comportamento de Compra

Aplicação de Machine Learning para segmentação de clientes utilizando o algoritmo **K-Means**, desenvolvida com Streamlit.

## Sobre o Projeto

Este projeto analisa padrões de compra de clientes de um atacadista e os agrupa em segmentos distintos com base em seu comportamento de consumo. A segmentação permite identificar perfis de clientes e direcionar estratégias de marketing e vendas.

## Dataset

Utilizamos o dataset **Wholesale Customers** do UCI Machine Learning Repository, contendo dados de gastos anuais de 440 clientes em 6 categorias de produtos:

| Variável         | Descrição                      |
| ---------------- | ------------------------------ |
| Fresh            | Gastos com produtos frescos    |
| Milk             | Gastos com laticínios          |
| Grocery          | Gastos com mercearia           |
| Frozen           | Gastos com congelados          |
| Detergents_Paper | Gastos com detergentes e papel |
| Delicassen       | Gastos com delicatessen        |

## Técnica Aplicada: K-Means

### O que é K-Means?

O **K-Means** é um algoritmo de aprendizado não supervisionado utilizado para **clusterização** (agrupamento). Ele particiona os dados em K grupos, onde cada observação pertence ao cluster com o centróide mais próximo.

### Como funciona?

1. **Inicialização**: Escolhe-se K centróides iniciais aleatoriamente
2. **Atribuição**: Cada ponto é atribuído ao centróide mais próximo
3. **Atualização**: Os centróides são recalculados como a média dos pontos do cluster
4. **Repetição**: Os passos 2 e 3 são repetidos até convergência

### Escolha do K (Método do Cotovelo)

Para determinar o número ideal de clusters, utilizamos o **Método do Cotovelo**:

- Calculamos a inércia (soma das distâncias ao quadrado dentro dos clusters) para diferentes valores de K
- O ponto onde a curva forma um "cotovelo" indica o K ideal
- Neste projeto, escolhemos **K = 4** clusters

### Métricas de Avaliação

- **Inércia**: Mede a compactação dos clusters (quanto menor, melhor)
- **Silhouette Score**: Mede a qualidade da separação entre clusters (-1 a 1, quanto maior, melhor)

## Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Treinar o modelo (opcional)

```bash
python src/train_model.py
```

### Executar a aplicação

```bash
python -m streamlit run app.py
# ou
streamlit run app.py
```

## Estrutura do Projeto

```
├── app.py                 # Aplicação Streamlit
├── requirements.txt       # Dependências
├── README.md              # Documentação
├── data/
│   ├── wholesale_customers.csv      # Dataset original
│   ├── wholesale_with_clusters.csv  # Dataset com clusters
│   ├── cluster_profile.csv          # Perfil médio dos clusters
│   └── elbow_plot.png               # Gráfico do cotovelo
├── models/
│   ├── kmeans_model.pkl   # Modelo treinado
│   └── scaler.pkl         # Normalizador
└── src/
    └── train_model.py     # Script de treinamento
```

## Resultados e Interpretações

O modelo identificou 4 perfis distintos de clientes:

- **Cluster 0: Pequenos estabelecimentos de alimentação**  
  Consumo moderado em Fresh, Grocery e Frozen, com valores baixos em Detergents_Paper e Delicassen.  
  Representa pequenos restaurantes, padarias, lanchonetes e negócios locais de alimentação.

- **Cluster 1: Grandes consumidores/distribuidores**  
  Maior volume geral entre os grupos, com Fresh, Milk e Detergents_Paper muito altos.  
  Perfil característico de grandes redes varejistas, distribuidores ou supermercados de grande porte.

- **Cluster 2: Varejo médio/grande com foco em Grocery e Frozen**  
  Gastos extremamente altos em Grocery e Frozen, superiores aos demais clusters.  
  Indica mercados médios/grandes ou estabelecimentos que demandam alto estoque.

- **Cluster 3: Pequenos varejistas/clientes de baixo porte**  
  Baixo consumo em todas as categorias, de forma relativamente uniforme.  
  Representa pequenos mercados de bairro, mercearias locais ou clientes de menor escala.

## Limitações

- Dataset relativamente pequeno (440 registros)
- Apenas 6 variáveis de consumo
- Não considera sazonalidade ou tendências temporais

## Possibilidades Futuras

- Incorporar mais variáveis (localização, frequência de compra)
- Testar outros algoritmos (DBSCAN, Hierarchical Clustering)
- Criar sistema de recomendação baseado nos clusters
