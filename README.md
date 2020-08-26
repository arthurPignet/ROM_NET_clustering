Research internship at Centre des Matériaux of MINES ParisTech 
==============================

Clustering Hyper-parameter optimization to build rom-net dictionary in solid mechanics
------------------------------

**author** : Arthur Pignet 

**contact** : arthur.pignet@mines-paristech.fr

*Part-time internship : Sept 2019 / Jan 2020*

Disclaimer: I cleaned up the code a bit before release. A few test runs indicate it still works but if you encounter problems please let me know, either here as issue or via email.



This is the code accompanying my internship at the Centre des Matériaux, of MINES ParisTech. 
In this intership, I study the construction of dictionaries of local reduced models by clustering
methods. By considering a set of pre-computed simulations’ solutions as points on a Grassmann manifold,
the non-linear space of solutions can be approximated by a set of affine subspaces, which are identified
by clustering on these points. The utility of the clustering is highly dependent on its hyper-parameters, especially
on its dissimilarity measure. As each measure induces its own clusters’ topology, the clustering
method must be adapted to the measure chosen. The internship was about dealing with a systemic methodology to optimize
the dictionary of local reduced models. It consists in encoding the pre-computed solutions into points on a
grassmann manifold, selecting a dissimilarity measure and the adapted clustering method, then computing
a local reduced model for each cluster. With a fixed number of modes, the use of a local reduced model
instead of a global one increases the accuracy by a factor of 1 to 3, depending on the hyper-parameters.

 

Project Organization
------------

    ├── LICENSE
    ├── README.md           <- The top-level README.
    │
    ├── \_can_be_deleted    <- Trash bin (!! git ignored)
    |
    ├── data                <- Contains two dataset. 
    |   ├── thermo-meca-result
    |   |           ├── dof_gloabl_G.npy    <- Numpy array containing the matrix of reduced coordinates from 1000 finite-elements simulations
    │   |           └── List_simu.npy       <- Numpy array containing the list of index delimitation between simulations
        └──      
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── dissimilarities.py  <- Definitions of dissimilarities tested
    ├── main.py             <- Main file, to gridsearch around the hyperparameters
    ├── Partition.py        <- Classes definition of partition and clusters
    └── postprocessing.py   <- To create some graphics
       


## Environment set up

Everything you need can be installed with pip, from the requirement.txt file at the root of the repository 
```bash
pip install -r $PATH/cartpole-requirement.txt
```

I would like to thank David RYCKELYNCK (Centre des matériaux) who allowed me to do
this internship on his project, and drove me during these months.