

import os

WORKING_DIR = "D:/Documents/Stages/S3R/paper/code"
DATA_DIR = WORKING_DIR + "data/result"
os.chdir(WORKING_DIR)

from Partition import *
from postprocessing import *
from dissimilarities import *


# public functions

def name(a):
    return [key for key, value in globals().items() if value is a][0]


def data_extraction(name_gamma_file, name_list_simu_file):
    # gamma = []  # matrix of reduced coordinates Q= V_g. gamma, where Q is the concatenation of Q_i
    # L = []  # list of index delimitation between Q_i's

    print("Extraction...")

    gamma = np.load(DATA_DIR + "\\" + name_gamma_file)  # read the reduced coordinates
    L = np.load(DATA_DIR + "\\" + name_list_simu_file)

    print('Reduced Coordinates Nxm, G ', gamma.shape)
    print('Number of samples ', gamma.shape[1])
    print('Nb Simulations ', len(L))
    k = 3
    print('Simulation data, number k=', k, ' ends at L[', k - 1, '] = ', L[k - 1])

    print("Nan in Gamma:" + str(np.isnan(gamma).any()))  # Looking for corrupted data

    return [gamma[:, :L[0]]] + [gamma[:, L[k - 1]:L[k]] for k in
                                range(1, len(L))]  # list of gamma_i matrices, such as Q_i =V_g gamma_i


# Main

# Loading

DIR_SAVE_PARTITION = "D:\\Documents\\stage\\S3R\\paper\\code\\Partitions"
Dir_save_figures = "D:\\Documents\\stage\\S3R\\paper\\code\\Figures"

"""
Data extraction

You can either extract the raw data and split them into train et test set, or just load pre-split sets.
In order to compare properly metrics, we suggest to use the same train et test sets for parallelize-computing.

"""
gamma = data_extraction("dof_global_G.npy", "List_simu.npy")
gamma_train, gamma_test = train_test_split(gamma, test_size=0.2)
with open("test_set", 'wb') as fichier:
    pk.dump(gamma_test, fichier)
    fichier.close()
with open("train_set", 'wb') as fichier:
    pk.dump(gamma_train, fichier)
    fichier.close()

# OR
with open("test_set", 'wb') as fichier:
    gamma_test = pk.load(fichier)
    fichier.close()
with open("train_set", 'wb') as fichier:
    gamma_train = pk.load(fichier)
    fichier.close()

""" parameters """
K = [3, 5, 7, 9, 12, 18]
clustering_functions = ["k_medoids", "spectral"]
DBSCAN_parameters = [1]
OPTICS_parameters = [10]
distances = [biprojection, schubert, grassmann, binet_cauchy, chordal, fubini_study, martin, procrustes]
Nb_modes = 2
nb_iter = 200

### (re)-initialization

Result_all = {
    distance: {"k_medoids": {"parameter": K,
                             'gain_DTAP': [[] for _ in K],
                             'gain_DTM': [[] for _ in K],
                             'gain_lowest_err': [[] for _ in K],
                             'mean_of_local_gain': [[] for _ in K]},
               "spectral": {"parameter": K,
                            'gain_DTAP': [[] for _ in K],
                            'gain_DTM': [[] for _ in K],
                            'gain_lowest_err': [[] for _ in K],
                            'mean_of_local_gain': [[] for _ in K]},
               "DBSCAN": {"parameter": DBSCAN_parameters,
                          'gain_DTAP': [[] for _ in DBSCAN_parameters],
                          'gain_DTM': [[] for _ in DBSCAN_parameters],
                          'gain_lowest_err': [[] for _ in DBSCAN_parameters],
                          'mean_of_local_gain': [[] for _ in DBSCAN_parameters]},
               "OPTICS": {"parameter": OPTICS_parameters,
                          'gain_DTAP': [[] for _ in OPTICS_parameters],
                          'gain_DTM': [[] for _ in OPTICS_parameters],
                          'gain_lowest_err': [[] for _ in OPTICS_parameters],
                          'mean_of_local_gain': [[] for _ in OPTICS_parameters]},
               } for distance in distances}
MATRICES_DIS = {key: None for key in distances}
average_result = []

# Computation and save of each combination required
time_tour = time.time()

for dist in distances:
    for clustering, dic in Result_all[dist].items():
        for i in range(len(dic['parameter'])):
            parameter = dic['parameter'][i]
            print("Dissimilarity: " + name(dist) + ", Method :" + clustering + " with " + str(
                parameter) + " as clustering parameter")
            if MATRICES_DIS[dist] is None:
                part = Partition(gamma_train, dist, clustering_parameter=parameter, clustering_method=clustering,
                                 bi_dim=False, nb_modes=Nb_modes)
                MATRICES_DIS[dist] = part.mat_dist
                listB = part.B

            part = Partition(gamma_train, dist, clustering_parameter=parameter, clustering_method=clustering, B=listB,
                             precomputed_dissimilarity_matrix=MATRICES_DIS[dist],
                             nb_modes=Nb_modes)
            for key, value in part.gains(gamma_test).items():
                Result_all[dist][clustering][key][i].append(value)

            os.chdir(DIR_SAVE_PARTITION)
            with open(name(dist) + "_" + clustering + "_" + str(parameter) + "_clusters.part", 'wb') as fichier:
                pk.dump(part, fichier)
                fichier.close()

with open("Result_all", 'wb') as fichier:
    pk.dump(Result_all, fichier)
    fichier.close()

print(time.time() - time_tour, " s for the first loop. The partitions object  have been saved.")
compt = 1

# Loop in order to average the results


while compt < nb_iter:
    compt += 1
    for dist in distances:
        for clustering, dic in Result_all[dist].items():
            for i in range(len(dic['parameter'])):
                parameter = dic['parameter'][i]
                print("Dissimilarity: " + name(dist) + ", Method :" + clustering + " with " + str(
                    parameter) + " as clustering parameter")
                if MATRICES_DIS[dist] is None:
                    part = Partition(gamma_train, dist, clustering_parameter=parameter, clustering_method=clustering,
                                     bi_dim=False, nb_modes=Nb_modes)
                    MATRICES_DIS[dist] = part.mat_dist
                    listB = part.B

                part = Partition(gamma_train, dist, clustering_parameter=parameter, clustering_method=clustering,
                                 B=listB,
                                 precomputed_dissimilarity_matrix=MATRICES_DIS[dist],
                                 nb_modes=Nb_modes)
                for key, value in part.gains(gamma_test).items():
                    Result_all[dist][clustering][key][i].append(value)

with open("Result_all", 'wb') as fichier:
    pk.dump(Result_all, fichier)
    fichier.close()

# post-processing
os.chdir(DIR_SAVE_PARTITION)
with open("Result_all", 'rb') as fichier:
    Result_all = pk.load(fichier)
    fichier.close()

diss_correlation(DIR_SAVE_PARTITION)
comparison_avg(DIR_SAVE_PARTITION)
comparison_max(DIR_SAVE_PARTITION)
comparison_std(DIR_SAVE_PARTITION)
