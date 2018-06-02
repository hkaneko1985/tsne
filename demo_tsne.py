# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of t-SNE (t-distributed Stochastic Neighbor Embedding) using scikit-learn

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_swiss_roll, make_s_curve
from sklearn.manifold import TSNE
import mpl_toolkits.mplot3d

data_flag = 1  # 1: s-curve dataset, 2: swiss-roll dataset
perplexity = 85  # 85 in data_flag = 1, 50 in data_flag = 2

number_of_samples = 1000
noise = 0
random_state_number = 100

if data_flag == 1:
    original_X, color = make_s_curve(number_of_samples, noise, random_state=0)
elif data_flag == 2:
    original_X, color = make_swiss_roll(number_of_samples, noise, random_state=0)

# plot
plt.rcParams["font.size"] = 18
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
p = ax.scatter(original_X[:, 0], original_X[:, 1], original_X[:, 2], c=color)
#fig.colorbar(p)
plt.tight_layout()
plt.show()

autoscaled_X = (original_X - original_X.mean(axis=0)) / original_X.std(axis=0, ddof=1)
Z = TSNE(perplexity=perplexity, n_components=2, init='pca',
         random_state=random_state_number).fit_transform(autoscaled_X)

# plot after tSNE
plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=color)
plt.xlabel("z1")
plt.ylabel("z2")
plt.tight_layout()
plt.show()
