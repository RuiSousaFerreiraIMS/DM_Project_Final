import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

#Clustering metrics (SS, SSB, SSW, R2)

##calculates SS:
def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.
    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.
    
    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    return ss

##calculates SSB:
def get_ssb(df, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.
    
    Returns:
    float: The between-group sum of squares of the DataFrame.
    """
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))))
    
    ssb = np.sum(ssb_i)
    return ssb

##calculates SSW:
def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.
    
    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats + [label_col]
    df_k = df[feats_label].groupby(by=label_col).apply(
        lambda col: get_ss(col, feats),
        include_groups=False
    )
    return df_k.sum()

##calculates R^2:
def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.
    
    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """
    df_sst_ = get_ss(df, feats)  # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)  # get ss within
    df_ssb_ = df_sst_ - df_ssw_  # get ss between
    # r2 = ssb/sst
    return (df_ssb_ / df_sst_)

#K selection

##tests different k values:
def get_r2_scores(df, feats, clusterer, min_k=1, max_k=9):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        df_concat = pd.concat([df,
                               pd.Series(labels, name='labels', index=df.index)], axis=1)
        r2_clust[n] = get_rsq(df_concat, feats, 'labels')
    return r2_clust



def visualize_silhouette_graf(df, range_clusters=[2, 3, 4, 5, 6]):
    
    # --- SETUP THE GRID ---
    # 2 Rows, 5 Columns = 10 slots available
    n_rows = 2
    n_cols = 5
    
    # Create the figure and array of axes
    # figsize is (width, height) - made it wide to fit 5 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
    
    # Flatten allows us to iterate through the grid as a simple list (0 to 9)
    axes_flat = axes.flatten() 

    # --- LOOP THROUGH CLUSTERS ---
    for i, nclus in enumerate(range_clusters):
        
        # Safety check: stop if we have more clusters than plots
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i] # Get the current subplot
        
        # 1. K-Means
        clusterer = KMeans(n_clusters=nclus, init='k-means++', n_init=10, random_state=42)
        cluster_labels = clusterer.fit_predict(df)

        # 2. Average Score 
        silhouette_avg = silhouette_score(df, cluster_labels)
        print(f"For n_clusters = {nclus}, the average score is: {silhouette_avg:.4f}")

        # 3. Setup the subplot (ax)
        ax.set_xlim([-0.1, 1])
        # The (nclus + 1) * 10 is to insert blank space between silhouette plots
        ax.set_ylim([0, len(df) + (nclus + 1) * 10])

        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for j in range(nclus):
            # Aggregate the silhouette scores for samples belonging to cluster j, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()

            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / nclus)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10 

        # 4. Titles and Labels
        ax.set_title(f"K = {nclus} (Avg: {silhouette_avg:.2f})", fontsize=11)
        ax.set_xlabel("Silhouette Coeff")
        ax.set_ylabel("Cluster Label")

        # Vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        # Remove y-ticks for cleaner look
        ax.set_yticks([]) 

    # --- CLEANUP ---
    # Hide any empty subplots (if you have fewer clusters than grid slots)
    for k in range(len(range_clusters), len(axes_flat)):
        axes_flat[k].axis('off')

    plt.tight_layout()
    plt.show()