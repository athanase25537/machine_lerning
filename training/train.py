import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Génération de données aléatoires
np.random.seed(42)
n_samples = 100
taille_maison = np.random.normal(100, 20, n_samples)
nombre_pieces = np.random.randint(2, 6, n_samples)
distance_centre = np.random.uniform(1, 20, n_samples)
prix = 50 + 2*taille_maison + 10*nombre_pieces - 3*distance_centre + np.random.normal(0, 20, n_samples)

# DataFrame simulé (pour l'affichage)
import pandas as pd
data = pd.DataFrame({
    'taille': taille_maison,
    'pieces': nombre_pieces,
    'distance': distance_centre,
    'prix': prix
})
print(data.head())

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data['taille'], data['pieces'], data['distance'], c=data['prix'], cmap='viridis')
ax.set_xlabel('Taille (m²)')
ax.set_ylabel('Nombre de pièces')
ax.set_zlabel('Distance centre (km)')
plt.colorbar(sc, label='Prix (k€)')
plt.title("Relation 3D entre les features et le prix")
plt.show()

import seaborn as sns
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()