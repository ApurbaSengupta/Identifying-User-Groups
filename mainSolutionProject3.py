"""
Created on Tue Nov 14 23:51:36 2017

@authors: Apurba Sengupta, Dhruv Desai

"""

import numpy as np
import gzip, time
import matplotlib.pyplot as plt
from scipy import spatial

start1 = time.time()

"""

Question 1.1

For the first part of the project, you are required to download the Yahoo user
click log dataset, which can be found at the following link: https://webscope.
sandbox.yahoo.com/catalog.php?datatype=r&did=49). The dataset contains over 45
million user visits to the Yahoo news articles, and clicks are binary features
indicating whether a user has clicked on a given article. The details of the 
data format are described in the Readme. The dataset is compressed into a 1.1 
GB zip file, which contains ten more zip files. For the first part of the 
project, you are only required to test using any one of the ten zip files, 
since the dataset is too large otherwise. Unzip the first part of the data, 
ydata-fp-td-clicks-v1 0.20090501.gz, and read it in to a matrix. It may be 
helpful to use Python's gzip library, which allows you to read data from zipped
files.

"""

print "\n Reading data into data matrix ...\n"

# create empty list to hold the data points
data_list = []

# open the archive file and read data from it
data_file = gzip.open('ydata-fp-td-clicks-v1_0.20090501.gz', 'r')

# extract the user features from each sample of the file
for line in data_file:
    data_list.append([st.split(':')[1] for st in line.split('|')[1].split(' ')[1:-2]])

# convert the lsit of data points into a data matrix
X = np.array(data_list).astype(float)

# store the number of rows and columns of the data matrix as constants
nrows, ncols = X.shape[0], X.shape[1]

# close the archive file
data_file.close()
    
end1 = time.time() - start1

print "\n Time taken to create data matrix from raw data = ", end1, "seconds\n"

"""

Question 1.2

In the next two parts of the project, you should implement an online version of
the k-means clustering algorithm. As noted in class, k-means is solving a 
non-convex optimization problem, for which there are many local minima. This 
means that the initialization of the cluster centroids can greatly affect the 
final result. In this part, implement both randomized centroid initialization,
as well as the k-means++ initialization algorithm discussed in class.

"""

num_centers = [5, 10, 25, 50, 100, 150, 230, 320, 410]

# set length of Markov chain
M = 100

# lists to hold mean, maximum and minimum distances for k-MC^2 initilalized centers and number of centers 
plot_mean = []
plot_max = []
plot_min = []
plot_k = []

# lists to hold mean, maximum and minimum distances for randomly initilalized centers and number of centers 
plot_mean_r = []
plot_max_r = []
plot_min_r = []
plot_k_r = []

# loop for different number of clusters
for K in num_centers:
    
    # *************************************************************************************
    
    # RANDOM INITIALIZATION OF CLUSTER CENTERS
    
    # *************************************************************************************
    
    start2 = time.time()
    
    # randomly initialize K centers
    CLIST_i_random = np.random.randint(0, nrows, K)
    
    # generate initial centers
    CLIST_random = np.array([X[c] for c in CLIST_i_random])
    
    
    # *************************************************************************************
     
    # k-MARKOV CHAIN MONTE CARLO or k-MC^2 INITIALIZATION OF CLUSTER CENTERS
    
    # *************************************************************************************
    
    # randomly initialize first center
    CLIST_i_kmcmc = [np.random.randint(0, nrows, 1)[0]]
    
    # set a unifrom proposal distribution Q
    Q = 1/float(nrows)
    
    # iterate over remaining (k - 1) clusters
    for k in range(1, K):
    
        # randomly choose a candidate data point according to Q
        x = np.random.randint(0, nrows, 1)[0]
    
        # get minimum distance among distances of this candidate point from all centers
        d_x = min([np.dot(X[x] - X[j], X[x] - X[j]) for j in CLIST_i_kmcmc])
        
        # iterate over to create Markov chain of length 100
        for m in range(M - 1):
        
            # randomly choose another data point according to Q
            y = np.random.randint(0, nrows, 1)[0]
        
            # get minimum distance among distances of this data point from all centers
            d_y = min([np.dot(X[y] - X[j], X[y] - X[j]) for j in CLIST_i_kmcmc])
           
            # compute acceptance probability
            acceptance_probability = np.minimum((d_y * Q)/(d_x * Q), 1)
            
            # check whether to accept data point or reject it 
            if acceptance_probability > np.random.random():
            
                # accept data point 
                x = y
                
        # create list of candidate initial centers
        CLIST_i_kmcmc.append(x)
    
    # generate initial centers
    CLIST_kmcmc = np.array([X[c] for c in CLIST_i_kmcmc])
    
    end2 = time.time() - start2
    
    print "\n Time taken to generate", K, "randomly initialized and k-mcmc initialized cluster centers = ", end2, "seconds\n"
    
    
    """

    Question 1.3

    Once the cluster centroids are chosen, the next step is to iteratively improve 
    the solution until a local optimum is achieved. As discussed in the lectures, 
    Lloyd's heuristic is one approach, but this requires taking a pass over the 
    entire dataset each time, which is not feasible for the dataset here. Instead,
    implement Mini-batch k-means, which will adjust the centroids by taking a 
    subset of the data each time. Generate plots showing the mean, minimum, and 
    maximum distance to the cluster centroids for the dataset, and run this for 
    several values of k, ranging from k = 5,..., 500. Which k has the lowest error?
    Using plots, compare the performance of the kmeans++ initialization and 
    randomized centroids.

    """

    # set number of iterations
    no_of_iterations = 100

    # set mini-batch size
    mini_batch_size = 100000

    start3 = time.time()
    
    # list for holding maximum distances for k-MC^2 and randomly initialized centers
    m_list = []
        
    m1_list = []
        
    # list fro holding number of iterations
    t_list = []
    
    # loop over the number of iterations
    for t in range(no_of_iterations):
        
        # generate random mini batches of the above set mini-batch size
        mini_batches = np.random.randint(0, nrows, mini_batch_size)
    
        X_mini_batch = X[mini_batches]
        
        
        # *******************************************************************************
        
        # RUNNING MINI-BATCH ONLINE k-MEANS WITH k-MC^2 INITIALIZED CENTERS
        
        # *******************************************************************************
        
        
        # create empty list to hold the list of data points assigned to a each center
        center_data_list = [[] for j in range(CLIST_kmcmc.shape[0])]
        
        # create empty list to hold the list of distances of data points assigned to a each center
        center_data_dist_list = [[] for j in range(CLIST_kmcmc.shape[0])]
        
        # distance matrix having distance of each data point to each center
        dist_matrix = spatial.distance_matrix(X_mini_batch, CLIST_kmcmc, p = 2)
        
        # get index of center assigned to each of the corresponding data point
        c_j_index = [np.argmin(dist) for dist in dist_matrix]
        
        # list of tuples of data point index and corresponding nearest center index         
        zipped1 = zip(c_j_index, np.arange(0, mini_batch_size))
        
        # list of distance between center and its assigned data point
        center_data_dist = [np.amin(di) for di in dist_matrix]
        
        count1 = 0
        
        # loop over all data points and corresponding centers
        for (k1, v1) in zipped1:
            
            count1 += 1
            
            # update center for every corrresponding data point
            CLIST_kmcmc[k1] = CLIST_kmcmc[k1] + np.divide((X_mini_batch[v1] - CLIST_kmcmc[k1]), count1)
            
            # put data points assigned to a particular center into the same list
            center_data_list[k1].append(v1)
            
            # put distances of data points assigned to a particular center into the same list
            center_data_dist_list[k1].append(center_data_dist[v1])
        
        # calculate maximum distance when k = 100        
        if K == 150:
            
            m = max([max(dist) for dist in center_data_dist_list if len(dist) > 0])
            
            m_list.append(m)
            t_list.append(t)
        
        # *******************************************************************************
        
        # RUNNING MINI-BATCH ONLINE k-MEANS WITH RANDOMLY INITIALIZED CENTERS
        
        # *******************************************************************************
        
        
        # create an empty list to hold the list of data points assigned to a each center
        center_data_list_r = [[] for j in range(CLIST_random.shape[0])]
        
        # create empty list to hold the list of distances of data points assigned to a each center
        center_data_dist_list_r = [[] for j in range(CLIST_kmcmc.shape[0])]
        
        # distance matrix having distance of each data point to each center
        dist_matrix_r = spatial.distance_matrix(X_mini_batch, CLIST_random, p = 2)
        
        # get index of center assigned to each of the corresponding data point
        c_j_index_r = [np.argmin(dist) for dist in dist_matrix_r]
                
        # list of tuples of data point index and corresponding nearest center index  
        zipped2 = zip(c_j_index_r, np.arange(0, mini_batch_size))
        
        # list of distance between center and its assigned data point
        center_data_dist_r = [np.amin(di) for di in dist_matrix_r]
        
        count2 = 0
        
        # loop over all data points and corresponding centers
        for (k2, v2) in zipped2:
            
            count2 += 1
            
            # center update step
            CLIST_random[k2] = CLIST_random[k2] + np.divide((X_mini_batch[v2] - CLIST_random[k2]), count2)
            
            # put data points assigned to a particular center into the same list
            center_data_list_r[k2].append(v2)
            
            # put distances of data points assigned to a particular center into the same list
            center_data_dist_list_r[k2].append(center_data_dist_r[v2])  
        
        # calculate maximum distance when k = 100
        if K == 150:
            
            m1 = max([max(dist) for dist in center_data_dist_list_r if len(dist) > 0])
            
            m1_list.append(m1)
    
    # plot maximum distances for k-MC^2 and randomly initialized centers when k = 100
    if K == 150:
        
        plt.xlabel("Number of iterations")
        plt.ylabel("Maximum Distance")
        plt.plot(t_list, m_list, label='maximum distance for k-MC^2 initialized centers')
        plt.plot(t_list, m1_list, label='maximum distance for randomly initialized centers')
        plt.legend()
        plt.show()    

    end3 = time.time() - start3

    print "\n Time taken to converge to", K, "clusters with randomly initialized and k-MC^2 initialized centers = ", end3, "seconds\n"
    
    start4 = time.time()
    
    # calculate mean, maximum and minimum distances of every data point from its respective center (k-MC^2 initilaized) and capture the mean values for each 

    mean_list = [np.mean(dist) for dist in center_data_dist_list if len(dist) > 0]
                                 
    max_list = [max(dist) for dist in center_data_dist_list if len(dist) > 0]
                                 
    min_list = [min(dist) for dist in center_data_dist_list if len(dist) > 0]     
       
    plot_mean.append(np.mean(mean_list))    
    plot_max.append(np.mean(max_list))
    plot_min.append(np.mean(min_list))
    plot_k.append(K)
    
    end4 = time.time() - start4
    
    print "\n Time taken to calculate the mean, maximum and minimum distances for both k-MC^2 initilaized and randomly initilaized centers = ", end4, "seconds\n"
    
plt.xlabel("Number of clusters 'k'")
plt.ylabel("Mean, Maximum and Minimum Distances")
plt.plot(plot_k, plot_mean, label='mean distance')
plt.plot(plot_k, plot_max, label='maximum distance')
plt.plot(plot_k, plot_min, label='minimum distance')
plt.legend()
plt.show()

end = time.time() - start1

print "\nTime taken by program to run = ", end, "seconds\n"