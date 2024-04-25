import numpy as np
import csv
from sklearn.cluster import KMeans
from pyransac3d import Plane
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def load_xyz_file(filename):
    points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            points.append([float(val) for val in row])
    return np.array(points)


def find_disjoint_clusters_kmeans(points, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(points)
    cluster_labels = kmeans.predict(points)
    unique_labels = np.unique(cluster_labels)

    clusters = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = points[cluster_labels == label]
        clusters.append(cluster)

    return clusters


def fit_plane_to_points_pyransac(points):
    plane = Plane()
    plane.fit(points)
    equation = plane.equation
    inliers = plane.inliers
    return equation, inliers


def fit_plane_to_points_dbscan(points, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points_scaled)
    unique_labels = np.unique(cluster_labels)

    clusters = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = points[cluster_labels == label]
        clusters.append(cluster)

    return clusters

filename = "punkty.xyz"
cloud_points = load_xyz_file(filename)
print("Wczytane punkty:")
print(cloud_points)

# Zadanie 3
num_clusters = 3
clusters_kmeans = find_disjoint_clusters_kmeans(cloud_points, num_clusters)
print("\nZadanie 3: Rozłączne chmury punktów za pomocą k-średnich:")
for i, cluster in enumerate(clusters_kmeans):
    print(f"Chmura punktów {i + 1}: {len(cluster)} punktów")

# Zadanie 4:
print("\nZadanie 4: Dopasowanie płaszczyzny:")
for i, cluster in enumerate(clusters_kmeans):
    equation, _ = fit_plane_to_points_pyransac(cluster)
    print(f"Dla klastra {i + 1}: Wektor normalny do płaszczyzny:", equation[:3])
    print("Płaszczyzna jest pionowa" if abs(equation[2]) > 0.95 else "Płaszczyzna jest pozioma")

# Zadanie 6:
print("\nZadanie 6: Powtórzenie zadania 3 i 4 z użyciem DBSCAN oraz pyransac3d:")
clusters_dbscan = fit_plane_to_points_dbscan(cloud_points)
if len(clusters_dbscan) == 0:
    print("Nie znaleziono klastrów za pomocą algorytmu DBSCAN.")
else:
    for i, cluster in enumerate(clusters_dbscan):
        equation, _ = fit_plane_to_points_pyransac(cluster)
        print(f"Dla klastra {i + 1}: Wektor normalny do płaszczyzny:", equation[:3])
        print("Płaszczyzna jest pionowa" if abs(equation[2]) > 0.95 else "Płaszczyzna jest pozioma")
