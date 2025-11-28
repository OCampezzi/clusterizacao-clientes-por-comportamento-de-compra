import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def treinar():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/wholesale_customers.csv")
    colunas = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    X = df[colunas]

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # gera grafico do cotovelo
    inercias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_norm)
        inercias.append(km.inertia_)

    plt.figure()
    plt.plot(range(2, 11), inercias, marker="o")
    plt.title("Método do Cotovelo")
    plt.xlabel("K")
    plt.ylabel("Inércia")
    plt.grid(True)
    plt.savefig("data/elbow_plot.png", bbox_inches="tight")
    plt.close()

    # treina modelo final com k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_norm)

    sil = silhouette_score(X_norm, df["cluster"])
    print(f"Silhouette Score: {sil:.4f}")

    # salva tudo
    df.groupby("cluster")[colunas].mean().to_csv("data/cluster_profile.csv")
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    df.to_csv("data/wholesale_with_clusters.csv", index=False)

    print("Modelo treinado e salvo!")

if __name__ == "__main__":
    treinar()
