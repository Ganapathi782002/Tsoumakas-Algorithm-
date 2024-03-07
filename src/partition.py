import numpy as np
from sklearn.cluster import KMeans
import utils


class PartitioningAlgorithm:

	def __init__(self):
		pass

	def partition(self,x_mat):
		pass


class RandomPartitioner(PartitioningAlgorithm):

	def __init__(self,num_partitions=2,partition_ratios=None):
		assert(num_partitions>1)
		self.num_partitions=num_partitions
		if not partition_ratios: 
			# ignore partition_ratios if passed and create balanced partition
			partition_ratios=np.zeros(num_partitions)+1
		else:
			assert(len(partition_ratios)==num_partitions)
			partition_ratios=partition_ratios
		cumulative_ratios=np.cumsum(partition_ratios)
		cumulative_ratios=cumulative_ratios/cumulative_ratios[-1]
		assert(cumulative_ratios[-1]==1)
		self.cumulative_ratios=cumulative_ratios

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		num_partitions=len(self.cumulative_ratios)
		# base case
		if num_partitions >= num_samples:
			padding = [[] for i in range(0,num_partitions-num_samples)]
			one_per_part=[[idx] for idx in range(0,num_samples)]
			return one_per_part + padding
		# generate a random permutation of the points
		permute=np.random.permutation(num_samples)
		# calculate the indices of each partition from cumulative_ratios
		partition_idcs=np.ceil(self.cumulative_ratios*num_samples).astype(dtype="int")
		# add the zero index
		partition_idcs=np.concatenate(([0],partition_idcs))
		assert(partition_idcs.shape[0]==num_partitions+1)
		partitions=[]
		for i in range(0,num_partitions):
			lpart=permute[partition_idcs[i]:partition_idcs[i+1]]
			partitions.append(lpart)
		return partitions


class KMeansPartitioner(PartitioningAlgorithm):

	def __init__(self,num_partitions=2,**kwargs):
		assert(num_partitions>1)
		self.num_partitions=num_partitions
		self.kmeanskwargs=kwargs

	def partition(self,x_mat):
		num_samples=x_mat.shape[0]
		assert(len(x_mat.shape)==2)
		# base case
		if self.num_partitions >= num_samples:
			padding = [[] for i in range(0,self.num_partitions-num_samples)]
			one_per_part=[[idx] for idx in range(0,num_samples)]
			return one_per_part + padding
		# run kmeans from scikit
		kmeans=KMeans(n_clusters=self.num_partitions, **self.kmeanskwargs)
		kmeans=kmeans.fit(x_mat)
		cluster_labels=kmeans.labels_
		# parse labels 
		partitions=[]
		for c in range(0,self.num_partitions):
			cluster_support=np.nonzero(cluster_labels==c)[0]
			partitions.append(cluster_support.tolist())
		return partitions


class BalancedKMeansPartitioner(PartitioningAlgorithm):
    
    def __init__(self, num_partitions=2, tol=0.003, max_iter=300):
        self.num_partitions = num_partitions
        self.tol = tol
        self.max_iter = max_iter

    def partition(self, x_mat):
        num_samples = x_mat.shape[0]
        assert x_mat.shape[0] == num_samples
        num_dims = x_mat.shape[1]
        # base case
        if self.num_partitions >= num_samples:
            padding = [[] for i in range(0, self.num_partitions - num_samples)]
            one_per_part = [[idx] for idx in range(0, num_samples)]
            return one_per_part + padding
        # maximum size of each balanced cluster
        max_size = np.ceil(num_samples / self.num_partitions)
        # initialize means
        permute = np.random.permutation(num_samples)
        means = x_mat[permute[:self.num_partitions], :]
        assert means.shape[0] == self.num_partitions
        # initialize loop vars
        new_means = np.zeros(means.shape)
        diff = np.inf
        num_iter = 0
        sorted_clusters = [None] * self.num_partitions
        # loop to convergence / limit
        while diff > self.tol and num_iter < self.max_iter:
            # calculate distances between means and labels
            dist = utils.calculate_euclidean_distances(x_mat, means)
            assert dist.shape == (num_samples, self.num_partitions)
            dist = np.sqrt(dist)

            # initialize sorted (w.r.t distance from mean)
            for i in range(0, self.num_partitions):
                sorted_clusters[i] = []
            # assign new labels keeping clusters balanced
            for idx in range(0, num_samples):
                finished = False
                # insert idx into its best cluster, possibly setting off a cascade to maintain
                # size of cluster < max_size
                ins_idx = idx
                while not finished:
                    best_cluster = np.argmin(dist[ins_idx, :])
                    sorted_clusters[best_cluster] = self._insert_in_sorted_list(sorted_clusters[best_cluster],
                                                                                 (ins_idx, dist[ins_idx, best_cluster]))
                    # check if size is still ok
                    if len(sorted_clusters[best_cluster]) <= max_size:
                        # done with the cascades (if any)
                        finished = True
                    else:
                        # remove the worst/last label from this cluster
                        ins_idx = sorted_clusters[best_cluster].pop(-1)[0]
                        dist[ins_idx, best_cluster] = np.inf

            # calculate new means
            for i in range(0, self.num_partitions):
                idcs = [el[0] for el in sorted_clusters[i]]
                new_means[i, :] = np.mean(x_mat[idcs, :], axis=0)
            # update
            diff = np.linalg.norm(means - new_means, ord=2)
            means = new_means
            num_iter = num_iter + 1
        partitions = [None] * self.num_partitions
        for i in range(0, len(sorted_clusters)):
            partitions[i] = [j for (j, dist) in sorted_clusters[i]]
        return partitions

    def _insert_in_sorted_list(self, sorted_list, element):
        # insert element into a sorted list of (sample_id,sorting_key) pairs
        _, sorting_key = element
        # dummy element for easier indexing
        sorted_list = [(-50, -np.inf)] + sorted_list
        # linear search for correct position
        for i in range(0, len(sorted_list)):
            # reached the last position
            if i == len(sorted_list) - 1:
                sorted_list.append(element)
                break
            if sorted_list[i][1] <= sorting_key and sorted_list[i + 1][1] > sorting_key:
                sorted_list = sorted_list[:i + 1] + [element] + sorted_list[i + 1:]
                inserted = True
                break
        sorted_list = sorted_list[1:]
        return sorted_list
