import matplotlib.pyplot as plt
import pandas as pd

# Lire les scores depuis le fichier CSV
scores = pd.read_csv("scoresbon.csv", header=None)

# print(scores)

plt.scatter(scores[0], scores[1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GHA - First Two Principal Components")
plt.show()
