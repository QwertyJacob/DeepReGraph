from DeepReGraph import *

# HOw to get the prmitive ccre PCA from a given gae object

y_true = list(nx.get_node_attributes(gae.G, 'primitive_cluster').values())[gae.ge_count:]
primitive_ccre_clusters = [int(prim_clust_srt.split('_')[1]) for prim_clust_srt in y_true]

# generate a list of markers and another of colors
plt.rcParams["figure.figsize"] = (15, 5)
for cluster in gae.ccre_ds['cluster'].unique():
    cluster_points = Z[primitive_ccre_clusters == cluster]
    plt.scatter(cluster_points[:, 0],
                cluster_points[:, 1],
                label='Cluster' + str(cluster))
plt.legend()
plt.title(
    'cCRE PCA 0 and 1: Explained Variance Ratio: [' + str(pca.explained_variance_ratio_[0]) + ','+ str(
        pca.explained_variance_ratio_[1]) + ']. \n   Singluar Values: [' + str(
        pca.singular_values_[0]) + ', '+ str(pca.singular_values_[1]) + ']' )

plt.show()

for cluster in gae.ccre_ds['cluster'].unique():
    cluster_points = Z[primitive_ccre_clusters == cluster]
    plt.scatter(cluster_points[:, 0],
                cluster_points[:, 2],
                label='Cluster' + str(cluster))
plt.legend()
plt.title(
    'cCRE PCA 0 and 2: Explained Variance Ratio: [' + str(pca.explained_variance_ratio_[0]) + ','+ str(
        pca.explained_variance_ratio_[2]) + ']. \n   Singluar Values: [' + str(
        pca.singular_values_[0]) + ', '+ str(pca.singular_values_[2]) + ']' )

plt.show()

for cluster in gae.ccre_ds['cluster'].unique():
    cluster_points = Z[primitive_ccre_clusters == cluster]
    plt.scatter(cluster_points[:, 1],
                cluster_points[:, 2],
                label='Cluster' + str(cluster))
plt.legend()
plt.title(
    'cCRE PCA 1 and 2: Explained Variance Ratio: [' + str(pca.explained_variance_ratio_[1]) + ','+ str(
        pca.explained_variance_ratio_[2]) + ']. \n   Singluar Values: [' + str(
        pca.singular_values_[1]) + ', '+ str(pca.singular_values_[2]) + ']' )

plt.show()