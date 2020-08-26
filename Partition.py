import time
from sklearn.cluster import SpectralClustering, OPTICS, DBSCAN

from sklearn import manifold
from sklearn.model_selection import train_test_split
from scipy import stats

from dissimilarities import *


# public functions

def name(a):
    return [key for key, value in globals().items() if value is a][0]


## Class

class Partition():

    def __init__(self, train_set, distance, clustering_parameter, clustering_method="k_medoids", nb_modes=2, B=None,
                 precomputed_dissimilarity_matrix=None, bi_dim=False):

        """
        Parameters:
            -train_set: list of data point (reduced coordonates) used for clustering
            -distance : distance function, set in the files "distances.py"
            -clustering_parameter : for k_medoids/spectal, int, Number of cluster asks to the clustering algorithm. As small clusters are going to trash, it can be higher than the final number of clusters.
                                    for optics : (int > 1 or float between 0 and 1)The number of samples in a neighborhood for a point to be considered as a core point.
                                    for DBSCAN : The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            -technique: string, "k_medoids" or "spectral". Set the clustering algorithm used.
            -nb_modes: set the number of modes for local reduced basis, and the modes's number of the global basis used for test.
            -B : None, or the list of precomputed reduced basis Bk.
            -precomputed_dissimilarity_matrix : None, or the precomputed dissimilarity matrix...
            -bi_dim: If True, 2 dimensional MDS will be computed during the initialisation




        """

        if nb_modes is None:
            self.nb_modes = train_set[0].shape[0]
        else:
            self.nb_modes = nb_modes

        self.data = train_set
        self.distance = distance
        self.clustering = clustering_method
        self.error_test_stat = None
        self.gain = None
        self.good_cluster_error = None
        self.wrong_cluster_error = None
        self.trash = []

        if not (B is None):
            self.B = B  # list of B_i
        else:
            start_init = time.time()
            print('Computation of Bk ...')
            self.B = self._listBk(1.e-5)
            print("Computed. " + str(round(time.time() - start_init, 3)) + " s")

        if not (precomputed_dissimilarity_matrix is None):
            self.mat_dist = precomputed_dissimilarity_matrix
        else:
            start_init = time.time()
            print('Computation of the dissimilarity matrix...')
            self.mat_dist = self._dissimilarity_matrix()
            print("Computed. " + str(round(time.time() - start_init, 3)) + " s")

        if self.clustering == "k_medoids":
            self.nb_cluster = clustering_parameter
            print("Clustering using K-medoid algorithm ...")
            start_init = time.time()
            self.clusters = self._k_medoids()
            for key in [key for key, value in self.clusters.items() if value.name == "trash"]:
                del self.clusters[key]
            print("Computed. " + str(round(time.time() - start_init, 3)) + " s")

        elif self.clustering == "spectral":
            self.nb_cluster = clustering_parameter
            print("Clustering using spectral algorithm ...")
            start_init = time.time()
            self.clusters = self._graph_cut()
            for key in [key for key, value in self.clusters.items() if value.name == "trash"]:
                del self.clusters[key]

            print("Computed. " + str(round(time.time() - start_init, 3)) + " s")
        elif self.clustering == "OPTICS":
            print("Clustering using OPTICS algorithm ...")
            start_init = time.time()
            self.clusters = self._optics()
            for key in [key for key, value in self.clusters.items() if value.name == "trash"]:
                del self.clusters[key]

            print("Clustering computed. " + str(round(time.time() - start_init, 3)) + " s")

        elif self.clustering == "DBSCAN":
            print("Clustering using DBSCAN algorithm ...")
            start_init = time.time()
            self.clusters = self._dbscan(clustering_parameter)
            for key in [key for key, value in self.clusters.items() if value.name == "trash"]:
                del self.clusters[key]

            print("Clustering computed. " + str(round(time.time() - start_init, 3)) + " s")

        else:
            print("No clustering techniques given")

        if bi_dim:
            print("Computation of MDS ...")
            start_init = time.time()
            self.embedded_coord, self.MDS_stress = self.MDS()
            print("Computed. " + str(round(time.time() - start_init, 3)) + " s")
        else:
            self.embedded_coord = None

    def MDS(self, NoMetric=False):
        Distances = self.mat_dist
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                           dissimilarity="precomputed", n_jobs=-1)
        pos = mds.fit(Distances).embedding_
        if NoMetric:
            nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                                dissimilarity="precomputed", n_jobs=-1,
                                n_init=1)
            point = nmds.fit_transform(Distances, init=pos)
            self.stress = nmds.stress_
        else:
            point = pos
            self.stress = mds.stress_
        return point, self.stress

    def predict_DTM(self, test_set):
        medoids = self._list_medoid()
        key_clusters = [key for key in self.clusters.keys()]
        distance_test_set_to_medoids = np.zeros((len(test_set), len(medoids)))
        test_predictions = [0 for i in test_set]
        for i in range(len(test_set)):
            temp_Bi_tested = self._calculBk(test_set[i])
            for j in range(len(medoids)):
                if self.distance == biprojection:
                    distance_test_set_to_medoids[i, j] = self.distance(test_set[i],
                                                                       np.dot(temp_Bi_tested.T, test_set[i]),
                                                                       self.B[medoids[j]],
                                                                       np.dot(self.B[medoids[j]].T,
                                                                              self.data[medoids[j]]))
                else:
                    distance_test_set_to_medoids[i, j] = self.distance(temp_Bi_tested, self.B[medoids[j]])

            test_predictions[i] = np.argmin(distance_test_set_to_medoids[i, :])
        return [key_clusters[test_predictions[i]] for i in range(len(test_set))]

    def predict_DTAP(self, test_set):
        distance_test_set_to_cluster = np.zeros((len(test_set), len(self.clusters.values())))
        key_clusters = [key for key in self.clusters.keys()]
        test_predictions = [0 for i in test_set]
        for i in range(len(test_set)):
            temp_Bi_tested = self._calculBk(test_set[i])
            for j in range(len(key_clusters)):
                clu = self.clusters[key_clusters[j]]
                temp_distance = np.zeros(len(clu.simulations))
                for k in range(len(clu.simulations)):
                    if self.distance == biprojection:
                        temp_distance[k] = self.distance(test_set[i], np.dot(temp_Bi_tested.T, test_set[i]),
                                                         self.B[clu.simulations[k]],
                                                         np.dot(self.B[clu.simulations[k]].T,
                                                                self.data[clu.simulations[k]]))
                    else:
                        temp_distance[k] = self.distance(temp_Bi_tested, self.B[clu.simulations[k]])
                distance_test_set_to_cluster[i, j] = np.nanmean(temp_distance)
            test_predictions[i] = np.argmin(distance_test_set_to_cluster[i, :])
        return [key_clusters[test_predictions[i]] for i in range(len(test_set))]

    def gains(self, test_set):
        """Use of DTLS_basis"""
        print("Computation of gains")
        started_time = time.time()
        error_proj_testset_in_DTLS_basis = np.zeros((len(test_set), len(self.clusters.items())))
        keys_clusters = [key for key in self.clusters.keys()]
        for i in range(len(test_set)):
            for j in range(len(self.clusters.keys())):
                clu = self.clusters[keys_clusters[j]]
                error_proj_testset_in_DTLS_basis[i, j] = np.linalg.norm(
                    test_set[i] - np.dot(clu.baseDTLS, np.dot(clu.baseDTLS.T, test_set[i]))) / np.linalg.norm(
                    test_set[i])

        error_global_base = np.zeros(len(test_set))
        print('Comparison with projection errors on a global basis with ', self.nb_modes, ' modes')
        for i in range(len(test_set)):
            error_global_base[i] = np.linalg.norm(test_set[i][self.nb_modes:, :]) / np.linalg.norm(test_set[i])

        lowest_error_predicted_cluster = np.zeros(error_proj_testset_in_DTLS_basis.shape[0]).astype(int)
        DTM_predicted_cluster = self.predict_DTM(test_set)
        DTAP_predicted_cluster = self.predict_DTAP(test_set)

        lowest_error_err = np.zeros(error_proj_testset_in_DTLS_basis.shape[0])
        DTM_err = np.zeros(error_proj_testset_in_DTLS_basis.shape[0])
        DTAP_err = np.zeros(error_proj_testset_in_DTLS_basis.shape[0])
        for i in range(error_proj_testset_in_DTLS_basis.shape[0]):
            lowest_error_predicted_cluster[i] = np.argmin(error_proj_testset_in_DTLS_basis[i, :])
            lowest_error_err[i] = error_proj_testset_in_DTLS_basis[i, lowest_error_predicted_cluster[i]]
            DTM_err[i] = error_proj_testset_in_DTLS_basis[i, keys_clusters.index(DTM_predicted_cluster[i])]
            DTAP_err[i] = error_proj_testset_in_DTLS_basis[i, keys_clusters.index(DTAP_predicted_cluster[i])]

        q_DTM = stats.mstats.mquantiles(DTM_err, 0.9)
        q_DTAP = stats.mstats.mquantiles(DTAP_err, 0.9)
        q_lowest_err = stats.mstats.mquantiles(lowest_error_err, 0.9)
        q_global_base = stats.mstats.mquantiles(error_global_base, 0.9)
        gain_DTM = q_global_base / q_DTM
        gain_DTAP = q_global_base / q_DTAP
        gain_lowest_err = q_global_base / q_lowest_err
        gain_mean_local_gain = np.mean([clu.local_gain() for clu in self.clusters.values()])
        print("gains computed: "+str(time.times()-started_time)+" s")
        return dict(gain_DTAP=gain_DTAP[0], gain_DTM=gain_DTM[0], gain_lowest_err=gain_lowest_err[0],
                    mean_of_local_gain=gain_mean_local_gain)

    def _list_medoid(self):
        return [cluster.medoid_id for cluster in self.clusters.values()]

    def _calculBk(self, matk,
                  tolerance_singular_value=10e-5):  # Computation of Bk in function of gamma and k. Note : Bk=V^T.Vk
        svd = np.linalg.svd(matk)
        return svd[0][:, [i for i in range(len(svd[1])) if svd[1][i] > svd[1][0] * tolerance_singular_value]]

    def _listBk(self, tolerance_singular_value=10e-5):
        return [self._calculBk(g, tolerance_singular_value) for g in self.data]

    def _dissimilarity_matrix(self):
        if self.distance == biprojection:
            L = self.B
            Distance = np.zeros((len(L), len(L)))
            for i in range(len(L)):
                for j in range(i):
                    d = self.distance(L[i], np.dot(L[i].T, self.data[i]), L[j], np.dot(L[j].T, self.data[j]))
                    Distance[i][j] = d
                    Distance[j][i] = d
        else:
            L = self.B
            Distance = np.zeros((len(L), len(L)))
            for i in range(len(L)):
                for j in range(i):
                    d = self.distance(L[i], L[j])
                    Distance[i][j] = d
                    Distance[j][i] = d
        return Distance

    def _k_medoids(self, itermax=200, randtest=5,
                   init=None):  # adapted form k_means algorithm, see https://mubaris.com/posts/kmeans-clustering/cd
        # random choice of initial medoids
        k = self.nb_cluster
        L = self.data
        Distances = self.mat_dist
        Best_result = {}, np.Infinity
        for i in range(randtest):
            if init is None:
                a = int(len(L) / k)
                C = [np.random.randint(i * a, (i + 1) * a) for i in range(k)]
                # C=[np.random.randint(0, len(L)-1) for i in range(k)]
                while 0 in Distances[np.ix_(C, C)] + np.eye(len(C),
                                                            len(C)):  # verification that medoids do not coincide
                    # C=np.array([np.random.randint(i*a, (i+1)*a) for i in range(k)])
                    C = [np.random.randint(0, len(L) - 1) for i in range(k)]

            else:
                C = init
            # print('Medoids initiaux'+str(C))
            C_old = np.array([0 for i in C])  # To store the index of medoids when it updates
            result = {}  # To store the list of indexs per cluster. Keys will correspond to the labels of clusters

            # Loop will stop if the error go to zero (ie if old medoids and new medoids coincide, for the distance used). If medoids do not converge, the loop will stop after itermax iteration, to avoid infinity loop
            for t in range(itermax):
                # determine clusters, i. e. arrays of data indices. Data points will be in the clusters with the nearest medoid
                J = np.argmin(Distances[:, C], axis=1)
                for kappa in range(k):
                    result[kappa] = np.where(J == kappa)[0]
                # update cluster medoids.
                C_old = np.copy(C)
                for kappa in range(k):
                    J = np.mean(Distances[np.ix_(result[kappa], result[kappa])], axis=1)
                    j = np.argmin(J)
                    C[kappa] = result[kappa][j]
                # check for convergence
                if np.array_equal(C, C_old):
                    # print("erreur nulle")
                    break

                # final update of cluster memberships
            else:
                # final update of cluster memberships
                J = np.argmin(Distances[:, C], axis=1)
                for kappa in range(k):
                    result[kappa] = np.where(J == kappa)[0]

            for i in [key for key, value in result.items() if value is []]:
                print(result.pop(i))
            ind = np.nanmean(
                [np.nanmean(Distances[np.ix_(l, l)][np.triu_indices(l.shape[0], k=1)]) for l in result.values()])
            if (ind < Best_result[1]):
                Best_result = result, ind

        return {key: (cluster(key, L, self, self.nb_modes)) for key, L in Best_result[0].items()}

    def _graph_cut(self, preExp=10):  # graphcut based on sklearn librairy
        k = self.nb_cluster
        Distances = self.mat_dist
        adjacency_matrix = preExp * np.exp(
            -(Distances / np.nanstd(Distances)) ** 2)  # compute the adjency matrix, from the dissimilarity matrix
        sc = SpectralClustering(k, affinity='precomputed', n_init=100, assign_labels='discretize')
        listcluster = sc.fit_predict(
            adjacency_matrix)  # list in which the element at the index k is the label of cluster, for the simulation k
        result = {}
        for i in range(k):
            result[i] = []
        for i in range(len(listcluster)):
            result[listcluster[i]].append(i)
        for i in [key for key, value in result.items() if len(value) == 0]:
            print(result.pop(i))

        return {key: (cluster(key, L, self, self.nb_modes)) for key, L in result.items()}

    def _optics(self, ):
        Distances = self.mat_dist
        op = OPTICS(min_samples=10, metric="precomputed")
        listcluster = op.fit(Distances).labels_  # list in which the element at the index k is the label of cluster,
        # for the simulation k
        result = {}
        self.nb_cluster = np.max(listcluster)
        print(self.nb_cluster)
        result[-1] = []
        for i in range(self.nb_cluster + 1):
            result[i] = []
        for i in range(len(listcluster)):
            result[listcluster[i]].append(i)
        for i in [key for key, value in result.items() if len(value) == 0]:
            print(result.pop(i))

        return {key: (cluster(key, L, self, self.nb_modes)) for key, L in result.items()}

    def _dbscan(self, eps):
        Distances = self.mat_dist
        op = DBSCAN(eps, min_samples=10, metric="precomputed")
        listcluster = op.fit(
            Distances).labels_  # list in which the element at the index k is the label of cluster, for the simulation k
        result = {}
        self.nb_cluster = np.max(listcluster)
        print(self.nb_cluster)
        result[-1] = []
        for i in range(self.nb_cluster + 1):
            result[i] = []
        for i in range(len(listcluster)):
            result[listcluster[i]].append(i)
        for i in [key for key, value in result.items() if len(value) == 0]:
            print(result.pop(i))

        return {key: (cluster(key, L, self, self.nb_modes)) for key, L in result.items()}


# noinspection DuplicatedCode,PyAttributeOutsideInit
class cluster():
    """
    Cluster
    Parameters:
        -name : name of the cluster
        -Simulations : List of data's index which were affected by the clustering at this cluster
        -partition : Parent of this cluster
        -nb_modes : Number of modes for the cluster's basis
        -baseDTLS : local reduced model computed by DTLS
        -baseDTLS_cv : local reduced model computed by DTLS cross validated


    """

    def __init__(self, name, simulations, partition, nb_modes):

        self.parent = partition
        self.name = str(name)
        self.simulations = simulations
        self.nb_modes = nb_modes

        if len(self.simulations) >= 10 and self.name != "-1":
            self.train_set, self.test_set = train_test_split(self.simulations, test_size=0.2)
            self.medoid_id = np.argmin(
                np.mean(self.parent.mat_dist[np.ix_(self.simulations, self.simulations)], axis=1))
            temp = self.DTLS_base()
            self.baseDTLS = temp[1]
            self.coordDTLS = temp[0]
        else:
            print("Moving the cluster " + str(name) + " into trash...")
            for i in self.simulations:
                self.parent.trash.append(i)
            self.name = "trash"

    def local_gain(self):
        local_err = np.zeros(len(self.test_set))
        global_err = np.zeros(len(self.test_set))
        for i in range(len(local_err)):
            local_err[i] = np.linalg.norm(
                self.parent.data[self.test_set[i]] - np.dot(self.baseDTLS, np.dot(self.baseDTLS.T, self.parent.data[
                    self.test_set[i]]))) / np.linalg.norm(
                self.parent.data[self.test_set[i]])
            global_err[i] = np.linalg.norm(
                self.parent.data[self.test_set[i]][self.parent.nb_modes:, :]) / np.linalg.norm(
                self.parent.data[self.test_set[i]])
        q_local_err = stats.mstats.mquantiles(local_err, 0.9)
        q_global = stats.mstats.mquantiles(global_err, 0.9)
        return q_global / q_local_err

    def _sorted_set(self, set):  # return a set of data sorted by there respective distances to medoid
        return sorted(set, key=lambda x: self.parent.mat_dist[self.medoid_id][x])

    def DTLS_base(self, tolerance_base=10e-2, tolerance_singular_value=10e-5):
        tronc = self.nb_modes
        sorted_train_set = self._sorted_set(self.train_set)
        self.learnt = [sorted_train_set[0], sorted_train_set[-1]]
        self.unlearnt = [i for i in sorted_train_set[1:]]
        Vg = self.parent.B[self.train_set[0]]
        Gg = np.dot(Vg.T, self.parent.data[sorted_train_set[0]])
        gamma_i = self.parent.data[sorted_train_set[-1]]
        # B_i = self.parent.B[sorted_train_set[-1]]
        # Qi=NU.gamma[i]=NU.BG.di+NU.Ri=NUn.Bg.di+NU.ui.ki
        di = np.dot(Vg.T, gamma_i)
        ri = gamma_i - np.dot(Vg, di)
        err = np.linalg.norm(ri) / np.linalg.norm(gamma_i)

        svd = np.linalg.svd(ri, compute_uv=True)
        # NU.Ri= NU.ui;sigmai.wi^T
        ui = svd[0][:, [i for i in range(len(svd[1])) if svd[1][i] > svd[1][0] * tolerance_singular_value]]
        ki = np.dot(ui.T, ri)
        Vg = np.concatenate((Vg, ui), axis=1)
        Gg = np.concatenate((Gg, di), axis=1)
        ki = np.concatenate((np.zeros((ki.shape[0], Gg.shape[1] - ki.shape[1])), ki), axis=1)
        Gg = np.concatenate((Gg, ki), axis=0)
        svd = np.linalg.svd(Gg, compute_uv=True)

        temp = svd[0][:, [i for i in range(len(svd[1])) if svd[1][i] > svd[1][0] * tolerance_singular_value]]
        Vg = np.dot(Vg, temp)
        Gg = np.dot(temp.T, Gg)

        while err > tolerance_base and len(self.unlearnt) > 0:
            a = self.parent.mat_dist[self.learnt, :][:, self.unlearnt]
            # print(a.shape)
            i = np.argmax(a) % np.shape(a)[1]
            self.learnt.append(self.unlearnt.pop(i))

            gamma_i = self.parent.data[i]
            # B_i = self.parent.B[i]
            # Qi=NU.gamma[i]=NU.BG.di+NU.Ri=NUn.Bg.di+NU.ui.ki
            di = np.dot(Vg.T, gamma_i)
            ri = gamma_i - np.dot(Vg, di)
            err = np.linalg.norm(ri) / np.linalg.norm(gamma_i)
            if err > tolerance_base:
                svd = np.linalg.svd(ri, compute_uv=True)
                # NU.Ri= NU.ui;sigmai.wi^T
                ui = svd[0][:, [i for i in range(len(svd[1])) if svd[1][i] > svd[1][0] * tolerance_singular_value]]
                ki = np.dot(ui.T, ri)
                Vg = np.concatenate((Vg, ui), axis=1)
                Gg = np.concatenate((Gg, di), axis=1)
                ki = np.concatenate((np.zeros((ki.shape[0], Gg.shape[1] - ki.shape[1])), ki), axis=1)
                Gg = np.concatenate((Gg, ki), axis=0)
                svd = np.linalg.svd(Gg, compute_uv=True)

                temp = svd[0][:, [i for i in range(len(svd[1])) if svd[1][i] > svd[1][0] * tolerance_singular_value]]
                Vg = np.dot(Vg, temp)
                Gg = np.dot(temp.T, Gg)

        return Gg[:tronc, :], Vg[:, :tronc]


## tests

if __name__ == "__main__":
    def data_extraction():
        #gamma is a matrix of reduced coordonates Q= V_g. gamma, where Q is the concatenation of Q_i
        #L is a list of index delimitations between Q_i's
        WORKING_DIR = "D:/Documents/Stages/S3R/paper/code"
        DATA_DIR = WORKING_DIR + "data/result"


        print("Extraction...")

        gamma = np.load(DATA_DIR + "/dof_global_G.npy")  # read the reduced coordinates
        L = np.load(DATA_DIR + "/List_simu.npy")

        print('Reduced Coordinates Nxm, G ', gamma.shape)
        print('Number of samples ', gamma.shape[1])
        print('Nb Simulations ', len(L))
        k = 3
        print('Simulation data, number k=', k, ' ends at L[', k - 1, '] = ', L[k - 1])

        print("Nan in Gamma:" + str(np.isnan(gamma).any()))  # Looking for corrupted data

        return [gamma[:, :L[0]]] + [gamma[:, L[k - 1]:L[k]] for k in
                                    range(1, len(L))]  # list of gamma_i matrices, such as Q_i =V_g gamma_i


    Dir_save_partition = "D:\\Documents\\S3R\\paper\\code\\Partitions2"
    Dir_save_figures = "D:\\Documents\\S3R\\paper\\code\\Figures"

    gamma = data_extraction()
    gamma, gamma_test = train_test_split(gamma, test_size=0.2)
    K = [3, 5, 7, 9, 12, 18]
    clustering_functions = ["k_medoids", "spectral", "DBSCAN", "OPTICS"]
    distances = [biprojection, schubert, grassmann, binet_cauchy, chordal, fubini_study, martin, procrustes]
    Nb_modes = 2

    chrono = time.time()
    temp = Partition(gamma, schubert, 0.01, clustering_method="OPTICS")
    print(temp.gains(gamma_test))
    print(time.time() - chrono)
