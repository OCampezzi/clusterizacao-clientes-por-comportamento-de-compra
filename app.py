import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st

st.set_page_config(
    page_title="Clusterização de Clientes",
    page_icon="🧩",
    layout="wide",
)

@st.cache_resource
def carregar_recursos():
    df = pd.read_csv("data/wholesale_with_clusters.csv")
    perfil = pd.read_csv("data/cluster_profile.csv", index_col=0)
    modelo = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, perfil, modelo, scaler

df, perfil_clusters, modelo_kmeans, scaler = carregar_recursos()

COLUNAS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

st.title("🧩 Agrupamento de Clientes por Comportamento de Compra")
st.markdown("""
Aplicação para analisar padrões de compra e segmentar clientes de acordo com seu perfil de consumo anual.
Modelo utilizado: **K-Means**.
""")

abas = st.tabs(["Visualização Geral", "Perfil dos Grupos", "Métricas e Técnica", "Simular Cliente"])

with abas[0]:
    st.subheader("Mapa 2D dos Clientes (PCA)")

    X = scaler.transform(df[COLUNAS])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": df["cluster"].astype(str)
    })

    fig, ax = plt.subplots()
    for c in sorted(df_plot["cluster"].unique()):
        grupo = df_plot[df_plot["cluster"] == c]
        ax.scatter(grupo["PC1"], grupo["PC2"], label=f"Cluster {c}", alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Amostra dos dados")
    st.dataframe(df[COLUNAS + ["cluster"]].head())

with abas[1]:
    st.subheader("Perfil Médio dos Clusters")
    st.dataframe(perfil_clusters.style.format("{:,.0f}"))

    st.markdown("""
### Interpretações
- Gastos altos em Grocery e Detergents_Paper: perfil de supermercados
- Foco em Fresh e Frozen: restaurantes ou food service
- Consumo reduzido e distribuído: varejistas menores
""")

with abas[2]:
    st.subheader("Métricas de Desempenho")

    X_scaled = scaler.transform(df[COLUNAS])
    sil_score = silhouette_score(X_scaled, df["cluster"])
    inercia = modelo_kmeans.inertia_

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters (K)", 4)
    col2.metric("Silhouette Score", f"{sil_score:.4f}")
    col3.metric("Inércia", f"{inercia:,.0f}")

    st.markdown("""
- **Silhouette Score**: varia de -1 a 1, quanto mais perto de 1, melhor a separação dos clusters
- **Inércia**: soma das distâncias quadradas dentro dos clusters, quanto menor melhor
""")

    st.divider()
    st.subheader("Método do Cotovelo")
    st.image("data/elbow_plot.png", caption="Gráfico do cotovelo para escolha do K")
    st.markdown("""
O método do cotovelo mostra a inércia para cada valor de K. Escolhemos o ponto onde a curva
começa a "achatar" - nesse caso, K=4 pareceu ser um bom valor.
""")

    st.divider()
    st.subheader("Sobre o K-Means")
    st.markdown("""
O K-Means é um algoritmo de clusterização não supervisionado. Ele funciona assim:

1. Escolhe K pontos iniciais (centróides)
2. Atribui cada dado ao centróide mais próximo
3. Recalcula os centróides baseado na média do grupo
4. Repete até estabilizar

É um algoritmo simples e rápido, mas precisa que a gente defina o K antes e pode ser afetado por outliers.
""")

with abas[3]:
    st.subheader("Classificar Novo Cliente")

    entradas = {}
    cols = st.columns(3)

    valores_padrao = {
        "Fresh": 12000, "Milk": 6000, "Grocery": 15000,
        "Frozen": 3000, "Detergents_Paper": 4000, "Delicassen": 2000,
    }

    for i, col in enumerate(COLUNAS):
        with cols[i % 3]:
            entradas[col] = st.number_input(col, min_value=0.0, value=float(valores_padrao[col]), step=500.0)

    if st.button("Classificar"):
        dados = np.array([entradas[c] for c in COLUNAS]).reshape(1, -1)
        dados_norm = scaler.transform(dados)
        cluster = int(modelo_kmeans.predict(dados_norm)[0])

        st.success(f"Cliente classificado no **Cluster {cluster}**")
        st.markdown("### Perfil médio desse grupo")
        st.dataframe(perfil_clusters.loc[[cluster]].style.format("{:,.0f}"))
