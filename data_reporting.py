import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def print_trends(original, min_cluster_size=0, max_cluster_size=200000):
    clusters = original.copy()

    datasets = [x for _, x in clusters.groupby('cluster')]

    plt.rcParams["figure.figsize"] = (10, 5)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i, ds in enumerate(datasets):

        if min_cluster_size <= ds.count()[0] <= max_cluster_size:

            print('cluster', i, 'len', ds.count()[0])

            '''
            ds['Heart_E10_5'] = ds['Heart_E10_5'] - ds['Heart_E10_5'].mean()
            ds['Heart_E11_5'] = ds['Heart_E11_5'] - ds['Heart_E11_5'].mean()
            ds['Heart_E12_5'] = ds['Heart_E12_5'] - ds['Heart_E12_5'].mean()
            ds['Heart_E13_5'] = ds['Heart_E13_5'] - ds['Heart_E13_5'].mean()
            ds['Heart_E14_5'] = ds['Heart_E14_5'] - ds['Heart_E14_5'].mean()
            ds['Heart_E15_5'] = ds['Heart_E15_5'] - ds['Heart_E15_5'].mean()
            ds['Heart_E16_5'] = ds['Heart_E16_5'] - ds['Heart_E16_5'].mean()
            ds['Heart_P0'] = ds['Heart_P0'] - ds['Heart_P0'].mean()
            '''

            sup_trend = np.array([pd.Series(ds['Heart_E10_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E11_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E12_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E13_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E14_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E15_5']).quantile(0.75),
                                  pd.Series(ds['Heart_E16_5']).quantile(0.75),
                                  pd.Series(ds['Heart_P0']).quantile(0.75)])
            inf_trend = np.array([pd.Series(ds['Heart_E10_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E11_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E12_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E13_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E14_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E15_5']).quantile(0.25),
                                  pd.Series(ds['Heart_E16_5']).quantile(0.25),
                                  pd.Series(ds['Heart_P0']).quantile(0.25)])

            for index, row in ds.iterrows():
                plt.plot(['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5', 'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5',
                          'Heart_E16_5', 'Heart_P0'],
                         row[['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5', 'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5',
                              'Heart_E16_5', 'Heart_P0']],
                         label=row[['EnsembleID']], color=colors[row['cluster'] % len(colors)], marker='o', alpha=0.1)

            plt.fill_between(
                x=['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5', 'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5',
                   'Heart_E16_5', 'Heart_P0'], y1=inf_trend, y2=sup_trend)

            plt.ylim((-2.5, 2.5))
            plt.show()


def print_ccre_trends(original, min_cluster_size=20, max_cluster_size=200):
    clusters = original.copy()

    datasets = [x for _, x in clusters.groupby('cluster')]

    plt.rcParams["figure.figsize"] = (30, 5)
    colors = ['y']

    for i, ds in enumerate(datasets):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('cluster ' + str(ds['cluster'].iloc[0]) + ' len ' + str(ds.count()[0]))
        stats = ds.describe()

        third_percentile = stats.loc['75%'].values.tolist()
        first_percentile = stats.loc['25%'].values.tolist()
        for index, row in ds.iterrows():
            clr_index = int(row['cluster'] % len(colors))

            ax1.plot(['E10_5_atac', 'E11_5_atac',
                      'E12_5_atac', 'E13_5_atac', 'E14_5_atac',
                      'E15_5_atac', 'E16_5_atac', 'P0_atac'], row[['Heart_E10_5_atac', 'Heart_E11_5_atac',
                                                                   'Heart_E12_5_atac', 'Heart_E13_5_atac',
                                                                   'Heart_E14_5_atac',
                                                                   'Heart_E15_5_atac', 'Heart_E16_5_atac',
                                                                   'Heart_P0_atac']], color=colors[clr_index],
                     marker='o', alpha=0.1)
            ax1.fill_between(x=['E10_5_atac', 'E11_5_atac',
                                'E12_5_atac', 'E13_5_atac', 'E14_5_atac',
                                'E15_5_atac', 'E16_5_atac', 'P0_atac'], y1=first_percentile[16:-2],
                             y2=third_percentile[16:-2], color='black')

            ax2.plot(['E10_5_acet',
                      'E11_5_acet', 'E12_5_acet', 'E13_5_acet',
                      'E14_5_acet', 'E15_5_acet', 'E16_5_acet',
                      'P0_acet'], row[['Heart_E10_5_acet',
                                       'Heart_E11_5_acet', 'Heart_E12_5_acet', 'Heart_E13_5_acet',
                                       'Heart_E14_5_acet', 'Heart_E15_5_acet', 'Heart_E16_5_acet',
                                       'Heart_P0_acet']], color=colors[clr_index], marker='o', alpha=0.1)
            ax2.fill_between(x=['E10_5_acet',
                                'E11_5_acet', 'E12_5_acet', 'E13_5_acet',
                                'E14_5_acet', 'E15_5_acet', 'E16_5_acet',
                                'P0_acet'], y1=first_percentile[8:16], y2=third_percentile[8:16], color='black')

            ax3.plot(['E10_5_met', 'E11_5_met', 'E12_5_met',
                      'E13_5_met', 'E14_5_met', 'E15_5_met',
                      'E16_5_met', 'P0_met'], row[['Heart_E10_5_met', 'Heart_E11_5_met', 'Heart_E12_5_met',
                                                   'Heart_E13_5_met', 'Heart_E14_5_met', 'Heart_E15_5_met',
                                                   'Heart_E16_5_met', 'Heart_P0_met']], color=colors[clr_index],
                     marker='o', alpha=0.1)
            ax3.fill_between(x=['E10_5_met', 'E11_5_met', 'E12_5_met',
                                'E13_5_met', 'E14_5_met', 'E15_5_met',
                                'E16_5_met', 'P0_met'], y1=first_percentile[0:8], y2=third_percentile[0:8],
                             color='black')

        plt.show()


def get_primitive_ccre_clusters(ccre_ds, primitive_ccre_path=''):
    ccre_agglomerative_ds = pd.read_csv(primitive_ccre_path+'agglomerative_clust_cCRE_8.csv')
    prim_ccre_ds = ccre_ds.set_index('cCRE_ID').join(ccre_agglomerative_ds.set_index('cCRE_ID'))[['cluster']]

    prim_ccre_ds.columns = ['primitive_cluster']
    return np.array(prim_ccre_ds.primitive_cluster.to_list())


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          ref='True label',
                          comp='Predicted label'):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel(ref)
        # plt.xlabel(comp + stats_text)
        plt.xlabel(comp)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


