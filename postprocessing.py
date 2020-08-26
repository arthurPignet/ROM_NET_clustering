from typing import List

import numpy as np
import pickle as pk
import time
import matplotlib.pyplot as plt
import os

from dissimilarities import *


def name(a):
    return [key for key, value in globals().items() if value is a][0]


def diss_correlation(Dir_save_partition):  # post-processing
    os.chdir(Dir_save_partition)
    with open("Result_all", 'rb') as fichier:
        Result_all = pk.load(fichier)
        fichier.close()

    # MATRICES_DIS[euclide] = MatDistance(listB,euclide)           #uncomment if you didn't compute these matrices before,else they are supposed to be stored in the global hashmap named M
    # MATRICES_DIS[grassmann] = MatDistance(listB,grassmann)
    # MATRICES_DIS[procrustes] = MatDistance(listB,procrustes)

    Schubert_list = []
    Euclide_list = []
    Grassmann_list = []

    for i in range(len(MATRICES_DIS[grassmann])):
        for j in range(i):
            Euclide_list.append(MATRICES_DIS[euclide][i, j])
            Grassmann_list.append(MATRICES_DIS[grassmann][i, j])

    plt.figure("Correlation between grassmann and euclidiean dissimilarities")
    plt.xlabel("Grassmann dissimilaritites")
    plt.ylabel("Euclidean dissimilaritites")
    plt.plot(Grassmann_list, Euclide_list, '.')

    plt.figure("Correlation between Schubert and Grassmann dissimilarities")
    plt.xlabel("Schubert dissimilaritites")
    plt.ylabel("Grassmann dissimilaritites")
    plt.plot(Schubert_list, Grassmann_list, '.')

    plt.figure("Schubert's dissimilarities distribution")
    plt.xlabel("Schubert's dissimilarities")
    plt.hist(Schubert_list, bins=400)

    plt.figure("Euclidian's dissimilarities distribution")
    plt.xlabel("Euclidian's dissimilarities")
    plt.hist(Euclide_list, bins=400)

    plt.figure("Grassmann's dissimilarities distribution")
    plt.xlabel("Grassmann's dissimilarities")
    plt.hist(Grassmann_list, bins=400)

    plt.show()


def MDS_repr(Dir_save_partition):
    os.chdir(Dir_save_partition)
    with open("Result_all", 'rb') as fichier:
        Result_all = pk.load(fichier)
        fichier.close()
    ### MDS representations
    with open("schubert_spectral_12_clusters.part", "rb") as fichier:
        part_schubert_spectral = pk.load(fichier)
        part_schubert_spectral.embedded_coord, part_schubert_spectral.MDS_stress = part_schubert_spectral.MDS()
        fichier.close()

    with open("schubert_k_medoids_12_clusters.part", "rb") as fichier:
        part_schubert_kmed = pk.load(fichier)
        part_schubert_kmed.embedded_coord, part_schubert_kmed.MDS_stress = part_schubert_spectral.embedded_coord, part_schubert_spectral.MDS_stress
        fichier.close()

    with open("biprojection_spectral_12_clusters.part", "rb") as fichier:
        part_biprojections_spectral = pk.load(fichier)
        part_biprojections_spectral.embedded_coord, part_biprojections_spectral.MDS_stress = part_biprojections_spectral.MDS()
        fichier.close()

    with open("biprojection_k_medoids_12_clusters.part", "rb") as fichier:
        part_biprojections_kmed = pk.load(fichier)
        part_biprojections_kmed.embedded_coord, part_biprojections_kmed.MDS_stress = part_biprojections_spectral.embedded_coord, part_biprojections_spectral.MDS_stress
        fichier.close()

    plt.figure("MDS plot of clustering results", figsize=(15, 8))

    plt.subplot(221)
    plt.title("MDS plot (from Schubert distances, spectral clustering)")
    for key, cluster in part_schubert_spectral.clusters.items():
        plt.scatter(part_schubert_spectral.embedded_coord[cluster.simulations, 0],
                    part_schubert_spectral.embedded_coord[cluster.simulations, 1], marker='.',
                    label="cluster " + str(key))
    if len(part_schubert_spectral.trash) > 0:
        plt.scatter(part_schubert_spectral.embedded_coord[part_schubert_spectral.trash, 0],
                    part_schubert_spectral.embedded_coord[part_schubert_spectral.trash, 1], marker='x', label="trash")
    plt.legend(loc="best")

    plt.subplot(222)
    plt.title("MDS plot (from Schubert distances, k-medoids clustering)")
    for key, cluster in part_schubert_kmed.clusters.items():
        plt.scatter(part_schubert_kmed.embedded_coord[cluster.simulations, 0],
                    part_schubert_kmed.embedded_coord[cluster.simulations, 1], marker='.', label="cluster " + str(key))
    if len(part_schubert_kmed.trash) > 0:
        plt.scatter(part_schubert_kmed.embedded_coord[part_schubert_kmed.trash, 0],
                    part_schubert_kmed.embedded_coord[part_schubert_kmed.trash, 1], marker='x', label="trash")
    plt.legend(loc="best")

    plt.subplot(223)
    plt.title("MDS plot (from biprojection distances, spectral clustering)")
    for key, cluster in part_biprojections_spectral.clusters.items():
        plt.scatter(part_biprojections_spectral.embedded_coord[cluster.simulations, 0],
                    part_biprojections_spectral.embedded_coord[cluster.simulations, 1], marker='.',
                    label="cluster " + str(key))
    if len(part_biprojections_spectral.trash) > 0:
        plt.scatter(part_biprojections_spectral.embedded_coord[part_biprojections_spectral.trash, 0],
                    part_biprojections_spectral.embedded_coord[part_biprojections_spectral.trash, 1], marker='x',
                    label="trash")
    plt.legend(loc="best")

    plt.subplot(224)
    plt.title("MDS plot (from biprojection distances, k-medoids clustering)")
    for key, cluster in part_biprojections_kmed.clusters.items():
        plt.scatter(part_biprojections_kmed.embedded_coord[cluster.simulations, 0],
                    part_biprojections_kmed.embedded_coord[cluster.simulations, 1], marker='.',
                    label="cluster " + str(key))
    if len(part_biprojections_kmed.trash) > 0:
        plt.scatter(part_biprojections_kmed.embedded_coord[part_biprojections_kmed.trash, 0],
                    part_biprojections_kmed.embedded_coord[part_biprojections_kmed.trash, 1], marker='x', label="trash")
    plt.legend(loc="best")
    plt.show()


def comparison_avg(Dir_save_partition):
    os.chdir(Dir_save_partition)
    with open("Result_all", 'rb') as fichier:
        Result_all = pk.load(fichier)
        fichier.close()
    ### Comparition

    F = plt.figure(
        "Influence of dissimilarity measure, of clustering technique, of K, and reduced base construction technique (average)",
        figsize=(15, 8))
    F.clear()
    #    plt.suptitle("Partition pour "+str(len(self.data))+' simulations, distance '+name(self.distance)+" methode "+self.clustering)
    # plt.suptitle("Gain depending on metrix, number of clusters, reduced-base constructions and clustering techniques")
    ax = plt.subplot(2, 2, 1)
    plt.title("K_medoids,  DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Average gain")
    for key, value in Result_all["k_medoids"]["DTLS_CV"].items():
        plt.plot(value['K'], np.mean(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 2)
    plt.title("Spectral, DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Average gain")
    for key, value in Result_all["spectral"]["DTLS_CV"].items():
        plt.plot(value['K'], np.mean(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 3)
    plt.title("K_medoids, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Average gain")
    for key, value in Result_all["k_medoids"]["DTLS"].items():
        plt.plot(value['K'], np.mean(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 4)
    plt.title("Spectral, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Average gain")
    for key, value in Result_all["spectral"]["DTLS"].items():
        plt.plot(value['K'], np.mean(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    plt.show()

    # comparison max


def comparison_max(Dir_save_partition):
    os.chdir(Dir_save_partition)
    with open("Result_all", 'rb') as fichier:
        Result_all = pk.load(fichier)
        fichier.close()
    F = plt.figure(
        "Influence of dissimilarity measure, of clustering technique, of K, and reduced base construction technique (max)",
        figsize=(15, 8))
    F.clear()
    #    plt.suptitle("Partition pour "+str(len(self.data))+' simulations, distance '+name(self.distance)+" methode "+self.clustering)
    # plt.suptitle("Gain depending on metrix, number of clusters, reduced-base constructions and clustering techniques")
    ax = plt.subplot(2, 2, 1)
    plt.title("K_medoids,  DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Maximal gain")
    for key, value in Result_all["k_medoids"]["DTLS_CV"].items():
        plt.plot(value['K'], np.max(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 2)
    plt.title("Spectral, DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Maximal gain")
    for key, value in Result_all["spectral"]["DTLS_CV"].items():
        plt.plot(value['K'], np.max(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 3)
    plt.title("K_medoids, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Maximal gain")
    for key, value in Result_all["k_medoids"]["DTLS"].items():
        plt.plot(value['K'], np.max(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 4)
    plt.title("Spectral, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Maximal gain")
    for key, value in Result_all["spectral"]["DTLS"].items():
        plt.plot(value['K'], np.max(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    plt.show()

    # comparison std


def comparison_std(Dir_save_partition):
    os.chdir(Dir_save_partition)
    with open("Result_all", 'rb') as fichier:
        Result_all = pk.load(fichier)
        fichier.close()
    F = plt.figure(
        "Influence of dissimilarity measure, of clustering technique, of K, and reduced base construction technique (standard deviation)",
        figsize=(15, 8))
    F.clear()
    #    plt.suptitle("Partition pour "+str(len(self.data))+' simulations, distance '+name(self.distance)+" methode "+self.clustering)
    # plt.suptitle("Gain depending on metrix, number of clusters, reduced-base constructions and clustering techniques")
    ax = plt.subplot(2, 2, 1)
    plt.title("K_medoids,  DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Standard deviation of Gain")
    for key, value in Result_all["k_medoids"]["DTLS_CV"].items():
        plt.plot(value['K'], np.std(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 2)
    plt.title("Spectral, DTLS cross validated reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Standard deviation of Gain")
    for key, value in Result_all["spectral"]["DTLS_CV"].items():
        plt.plot(value['K'], np.std(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 3)
    plt.title("K_medoids, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Standard deviation of Gain")
    for key, value in Result_all["k_medoids"]["DTLS"].items():
        plt.plot(value['K'], np.std(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    ax = plt.subplot(2, 2, 4)
    plt.title("Spectral, DTLS reduced basis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Standard deviation of Gain")
    for key, value in Result_all["spectral"]["DTLS"].items():
        plt.plot(value['K'], np.std(value['gain'], axis=1), '-x', label=name(key))
    plt.legend(loc='best')

    plt.show()

    """
    
    
    ### utility of clustering
    partition_tested = Partition(gamma, biprojection, 10, technique = "spectral")
    partition_tested_2 = Partition(gamma, schubert, 15)
    os.chdir(Dir_save_partition+"\\..")
    with open("test_set", 'rb') as fichier:
        test_set=pk.load(fichier)
        fichier.close()
    for i in partition_tested.clusters.keys():
        plt.figure()
        basis = partition_tested.clusters[i].baseDTLS_cv
        medoid_id = partition_tested.clusters[i].medoid_id
    
        for key, clu in partition_tested.clusters.items():
            err_proj = [np.linalg.norm(q - np.dot( basis[:,:Nb_modes], np.dot(basis[:,:Nb_modes].T, q)))/np.linalg.norm(q) for q in test_set]
            dist =  partition_tested.mat_dist[medoid_id][clu.simulations]
            plt.plot( dist,err_proj, '.', label="cluster "+str(key))
        plt.legend(loc="best")
        plt.xlabel("Dissimilarity")
        plt.ylabel("Projection error")
        plt.show()
    
    
    for i in partition_tested_2.clusters.keys():
        plt.figure()
        basis = partition_tested_2.clusters[i].baseDTLS_cv
        medoid_id = partition_tested_2.clusters[i].medoid_id
        for key, clu in partition_tested_2.clusters.items():
            err_proj = [np.linalg.norm(q - np.dot( basis[:,:Nb_modes], np.dot(basis[:,:Nb_modes].T, q)))/np.linalg.norm(q) for q in [partition_tested_2.data[i] for i in clu.simulations]]
            dist =  partition_tested_2.mat_dist[medoid_id][clu.simulations]
            plt.plot( dist,err_proj, '.', label="cluster "+str(key))
        plt.legend(loc="best")
        plt.xlabel("Dissimilarity")
        plt.ylabel("Projection error")
        plt.show()
    
    #gain_distance = [np.linalg.norm(q[:,:Nb_modes])/np.linalg.norm(q - np.dot( basis[:,:Nb_modes], np.dot(basis[:,:Nb_modes].T, q))) for [partition_tested.data[i] for i in clu.simulations] ]
    #diss_basis = partition_tested.mat_dist[medoid_id][test_set]
    
    
    F=plt.figure("Utility of clustering", figsize=(15,8))
    
    plt.subplot(131)
    plt.plot(diss_train_set,err_train_set, '.')
    plt.xlabel("Dissimilarity")
    plt.ylabel("Projection error")
    plt.title("Correlation between projection error and dissimilarity on the train set")
    
    plt.subplot(132)
    plt.plot(diss_test_set, err_test_set, '.')
    plt.xlabel("Dissimilarity")
    plt.ylabel("Projection error")
    plt.title("Correlation between projection error and dissimilarity on the test set")
    
    plt.subplot(133)
    plt.plot(diss_basis, gain_distance, '.')
    plt.xlabel("Dissimilarity")
    plt.ylabel("Gain")
    plt.title("Gain function of dissimilarity with the local reduced model")
    
    plt.show()"""
