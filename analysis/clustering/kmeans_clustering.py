
import numpy as np

from scipy.spatial.distance import cdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def run_kmeans(clusters_range, df, n_runs):
    
    meanDistortions = []
    sil_score = []
    inertias = []
    
    for i in range(n_runs):
        run_sil = []
        run_dist = []
        run_in = []
        for k in clusters_range:
            model = KMeans(n_clusters=k, random_state=i)
            model.fit(df)
            prediction = model.predict(df)
            score = silhouette_score(df, prediction)
            distortion = (
                sum(
                    np.min(cdist(df, model.cluster_centers_, "euclidean"), axis=1)
                )
                / df.shape[0]
            )
            run_in.append(model.inertia_)
            run_dist.append(distortion)
            run_sil.append(score)
        sil_score.append(run_sil)
        meanDistortions.append(run_dist)
        inertias.append(run_in)
    return meanDistortions, sil_score, inertias