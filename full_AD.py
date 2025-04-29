import os
import pickle
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from collections import OrderedDict
from taxonomy import get_classification_labels, get_astrophysical_class, plot_colored_tree, source_node_label, get_taxonomy_tree
import tensorflow as tf
from tensorflow import keras
import itertools

from LSST_Source import LSST_Source
from dataloader import LSSTSourceDataSet
from taxonomy import get_taxonomy_tree

import random
from interpret_results import save_all_cf_and_rocs, save_leaf_cf_and_rocs, get_conditional_probabilites
from dataloader import augment_ts_length_to_days_since_trigger, get_ts_upto_days_since_trigger, get_augmented_data, get_static_features
from vizualizations import plot_confusion_matrix
import networkx as nx
import gc
import imageio
from matplotlib import animation

days = 2 ** np.array(range(11))

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save(save_path , obj):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def make_gif(files, gif_file=None):

    # Load the images
    images = []
    for filename in files:
        images.append(imageio.imread(filename))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(18, 18))

    # Create the animation
    def animate(i):
        ax.clear()
        ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(images[i])

    fig.tight_layout()
        
    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=500)

    if gif_file:
        # Save the animation as a GIF
        anim.save(gif_file)

class AnomalyDetectionORACLE:
    def __init__(self):
        # @TODO: add options to change these attributes through default init parameters
        self._class_count = 10
        self._ts_length = 500
        self._ts_flag_value = 0.

        self._model = keras.models.load_model(f"models/lsst_alpha_0.5/best_model.h5", compile=False)
        self._model_dir = 'plots/testing/gif'
        self._tree = get_taxonomy_tree()

        self._default_seed = 42
        self._default_batch_size = 1024

        self._static_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'MW_plane_flag', 'ELAIS_S1_flag', 'XMM-LSS_flag', 'Extended_Chandra_Deep_Field-South_flag', 'COSMOS_flag']

        self._max_class_count = 10000000000

    def load_data(self):
        # @TODO: add options to change the path to the pickles dir with default parameter
        self._X_ts = load(f"../../oracle_code/ELAsTiCC-Classification/pickles/x_ts.pkl")
        self._X_static = load(f"../../oracle_code/ELAsTiCC-Classification/pickles/x_static.pkl")
        self._Y = load(f"../../oracle_code/ELAsTiCC-Classification/pickles/y.pkl")
        self._astrophysical_classes = load(f"../../oracle_code/ELAsTiCC-Classification/pickles/a_labels.pkl")

        for i in tqdm(range(len(self._X_static))):        
            self._X_static[i] = self._get_static_features(self._X_static[i])
        
        ellip_index = self._static_feature_list.index('HOSTGAL_ELLIPTICITY')
        print([self._X_static[i][ellip_index] for i in range(50)])

    def _get_static_features(self, y):
        self._missing_data_flags = [-9, -99, -999, -9999, 999]
        self._static_flag_value = -9
        self._ts_flag_value = 0

        static_features = []

        # Get the necessary static features from the ordered dictionary
        for feature in self._static_feature_list:
    
            if feature == 'HOSTGAL_ELLIPTICITY':
                static_features.append(-9)
                continue
            
            val = y[feature]
            if val in self._missing_data_flags:
                static_features.append(self._static_flag_value)
            else:
                static_features.append(val)

        return static_features

    def _determine_anomalies(self, class_probs):
        purity_threshold = 0.7
        print(f'PURITY : {purity_threshold}')
        # takes model output and converts to binary anomaly detection/non-detection
        level_order_nodes = list(nx.bfs_tree(self._tree, source=source_node_label).nodes())
        leaf_nodes = level_order_nodes[-19:]

        indiv_probs = class_probs[0]
        class_probs_df = pd.DataFrame(np.squeeze(indiv_probs))
        
        columns = level_order_nodes.copy()
        class_probs_df.columns = columns
          
        # groups nodes for anomaly detection
        anomaly_labels = class_probs_df[['KN', 'uLens', 'SLSN', 'PISN', 'TDE', 'CART', 'ILOT']]
        non_anomaly_labels = class_probs_df[['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB', 'SNIa', 'SNIax', 'SNIb/c', 'SNI91bg', 'SNII', 'M-dwarf Flare', 'Dwarf Novae', 'AGN']]
        anomaly_preds = [np.sum(lightcurve[1][::]) for lightcurve in anomaly_labels.iterrows()]
        non_anomaly_preds = [np.sum(lightcurve[1][::]) for lightcurve in non_anomaly_labels.iterrows()]
        anomaly_detections = list(enumerate(zip(anomaly_preds, non_anomaly_preds)))
    
        # maps class labels to generate y_true for anomalies
        # labels = [class_probs[i][1] for i in range(100)]
        labels = np.squeeze(class_probs[1])
        print(f'LEN LABELS: {len(labels)}')
    
        # true_class_names = []
        # level_order_nodes = list(nx.bfs_tree(tree, source=source_node_label).nodes())
        # leaf_nodes = level_order_nodes[-19:]
    
        # for class_idx in labels:
            # true_class_names.append(leaf_nodes[np.argmax(class_idx[-19:])])
    
        # class_df = pd.DataFrame(true_class_names, columns=['True Class:'])
        # df = pd.concat([class_df, class_probs_df], axis=1, ignore_index=True)
    
        # columns = level_order_nodes.copy()
        # columns.insert(0, 'True Class:')
        # df.columns = columns
    
        anomaly_map = [
            0.0, # AGN - not anomaly
            0.0, # SNIa - not anomaly
            0.0, # SNIb/c - not anomaly
            0.0, # SNIax - not anomaly
            1.0, # SNI91bg - anomaly
            0.0, # SNII - not anomaly
            1.0, # KN - anomaly
            0.0, # Dwarf Novae - not anomaly
            1.0, # uLens - anomaly
            0.0, # M-dwarf Flare - not anomaly
            1.0, # SLSN - anomaly
            1.0, # TDE - anomaly
            1.0, # ILOT - anomaly
            1.0, # CART - anomaly
            1.0, # PISN - anomaly
            0.0, # Cepheid - not anomaly
            0.0, # RR Lyrae - not anomaly
            0.0, # Delta Scuti - not anomaly
            0.0  # EB - not anomaly
        ]
        # for i, label in enumerate(labels):
        #     print(label)
        #     print(np.argmax(label))
        #     if i == 2:
        #         break
            
        y_true = [[anomaly_map[np.argmax(label[-19:])], 1.0 - anomaly_map[np.argmax(label[-19:])]] for label in labels]
        # print(f'LEN Y_TRUE: {len(y_true)}')
        # print(list(zip(y_true[:10], astrophysical_classes[:10])))
    
        # reformats anomaly detections to match y_true for metric calculation
        anomaly = np.array([anomaly_detections[row][1][0] for row in range(len(anomaly_detections))])
        non_anomaly = np.array([anomaly_detections[row][1][1] for row in range(len(anomaly_detections))])
        y_pred = np.stack((anomaly, non_anomaly), axis=1)
    
        # converts from one hot encoding to argmax representation for sklearn.metrics functions
        true = [np.argmax(y_true[i]) for i in range(len(y_true))]
        # print(f'LEN TRUE: {len(true)}')
        pred = [0 if element > purity_threshold else 1 for element in np.transpose(y_pred)[0]]

        return true, pred
    
    def run_anomaly_detection(self, purity_threshold=0.7):
        # @TODO: add parameter to specify whether to load in data from saved pickles or re-augment
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        all_predictions = []
        all_trues = []
        
        for d in days:
            
            print(f'Running inference for trigger + {d} days...')
    
            x1, x2, y_true, _ = augment_ts_length_to_days_since_trigger(self._X_ts, self._X_static, self._Y, self._astrophysical_classes, d)
            print('Data augmented, running prediction...')
            # save(f"pickles/augmented/day{d}/x1.pkl", x1)
            # save(f"pickles/augmented/day{d}/x2.pkl", x2)
            # save(f"pickles/augmented/day{d}/y.pkl", y_true)
            # y_true = load(f'pickles/augmented/day{d}/y.pkl')
            
            # Run inference on these
            y_pred = self._model.predict([x1, x2], batch_size=self._default_batch_size)
            print('Predicted!')
            # save(f"pickles/augmented/day{d}/pred.pkl", y_pred)
            # y_pred = load(f"pickles/augmented/day{d}/pred.pkl")
            print(f'LEN y_true: {len(y_true)}')
            print(f'LEN y_pred: {len(y_pred)}')
            
            # Get the conditional probabilities
            _, pseudo_conditional_probabilities = get_conditional_probabilites(y_pred, self._tree)
            
            print(f'For trigger + {d} days, these are the statistics:')
                
            plot_title = f"Trigger + {d} days, {purity_threshold * 100}% confidence threshold"
            
            # Print all the stats and make plots...
            # save_all_cf_and_rocs(y_true, pseudo_conditional_probabilities, tree, model_dir, plot_title)
            save_leaf_cf_and_rocs(y_true, pseudo_conditional_probabilities, self._tree, self._model_dir, plot_title)
            probs_with_labels = [[pseudo_conditional_probabilities[::]], [y_true[::]]]

            print('Determining anomalies...')
            print(purity_threshold)
            true, pred = self._determine_anomalies(probs_with_labels)
    
            labels = ['Anomaly', 'Not Anomaly']
            for index, (label, prediction) in enumerate(zip(true, pred)):
                true[index] = labels[label]
                pred[index] = labels[prediction]

            print('Plotting confusion matrix...')
            # print(list(zip(true, pred))[0:5])
            plot_confusion_matrix(true, pred, labels, title=plot_title, img_file=f'plots/testing/AD_{d}.png')
            
            all_predictions.append(pseudo_conditional_probabilities)
            all_trues.append(y_true)
    
            plt.close()
    
        # Make the gifs at leaf nodes for days
        cf_files = [f"{self._model_dir}/gif/leaf_cf/Trigger + {d} days.png" for d in days]
        make_gif(cf_files, f'{self._model_dir}/gif/leaf_cf/leaf_cf_days.gif')
        plt.close()

        roc_files = [f"{self._model_dir}/gif/leaf_roc/Trigger + {d} days.png" for d in days]
        make_gif(roc_files, f'{self._model_dir}/gif/leaf_roc/leaf_roc_days.gif')
        plt.close()

        # Make the gifs at the level 1 of the tree
        cf_files = [f"{self._model_dir}/gif/level_1_cf/Trigger + {d} days.png" for d in days]
        make_gif(cf_files, f'{self._model_dir}/gif/level_1_cf/level_1_cf_days.gif')
        plt.close()

        roc_files = [f"{self._model_dir}/gif/level_1_roc/Trigger + {d} days.png" for d in days]
        make_gif(roc_files, f'{self._model_dir}/gif/level_1_roc/level_1_roc_days.gif')
        plt.close()

        # Make the gifs at the level 2 of the tree
        cf_files = [f"{self._model_dir}/gif/level_2_cf/Trigger + {d} days.png" for d in days]
        make_gif(cf_files, f'{self._model_dir}/gif/level_2_cf/level_2_cf_days.gif')
        plt.close()

        roc_files = [f"{self._model_dir}/gif/level_2_roc/Trigger + {d} days.png" for d in days]
        make_gif(roc_files, f'{self._model_dir}/gif/level_2_roc/level_2_roc_days.gif')

    def test(self):
        random.seed(self._default_seed)

        a, b = np.unique(self._astrophysical_classes, return_counts=True)
        print(f"Total sample count = {np.sum(b)}")
        print(pd.DataFrame(data = {'Class': a, 'Count': b}))

        X_ts_balanced = []
        X_static_balanced = []
        Y_balanced = []
        astrophysical_classes_balanced = []

        for c in np.unique(self._astrophysical_classes):
    
            idx = list(np.where(np.array(self._astrophysical_classes) == c)[0])
            
            if len(idx) > self._max_class_count:
                idx = random.sample(idx, self._max_class_count)
        
            X_ts_balanced += [self._X_ts[i] for i in idx]
            X_static_balanced += [self._X_static[i] for i in idx]
            Y_balanced += [self._Y[i] for i in idx]
            astrophysical_classes_balanced += [self._astrophysical_classes[i] for i in idx]

        a, b = np.unique(astrophysical_classes_balanced, return_counts=True)
        data_summary = pd.DataFrame(data = {'Class': a, 'Count': b})
        data_summary.to_csv(f"{self._model_dir}/test_sample.csv")
        print(data_summary)

        self.run_anomaly_detection()

if __name__ == '__main__':
    ad = AnomalyDetectionORACLE()
    ad.load_data()
    ad.test()
