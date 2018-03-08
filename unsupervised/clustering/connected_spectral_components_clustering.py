import numpy as np
from graph.queue import Queue
from PIL import Image

def cluster(X, similarity_metric, thresh_dist, n_steps = 50):
    A = np.zeros((X.shape[0], X.shape[0]), dtype = np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i,j] = similarity_metric(X[j], X[i])
    #A[A > 1] = 1000
    #raising Affinity matrix A^N yields a matrix A' where A'[i,j] is the sum of affinities of all possible walks to get from i to j'
    #This means that if A'[i,j] is comparably low to the rest of A', then A[i,j] is a member of a dense set of points/is on it's own and is "difficult" to
    #reach from far away points (for example if it is off in it's own connected component)

    #the MINIMUM sum of affinities of all walks to get from i to j is achieved by multiplying A by itself N times in a different fashion:
    #let A' = A^N+1, if A'[i,j] = sum of all walks in N+1 steps that start with i and end with j = sum of all walks in 1 step from walks of N steps that began with i and end with j
    #want min sum to be A' = min walk in 1 step ending with j from min walks of N steps that begin with i and end with :
    print("A: ", A)
    #A_walk = __min_walks(A, n_steps, min_dists = False)
    #print("A_walk: ", A_walk)

    clusters = __connected_component_cluster(A, thresh_dist)
    #img = Image.fromarray(np.uint8(255*A/A.max()))
    #img.show()

    return clusters

def __walk(A, N):
    prod = np.identity(A.shape[0])
    for i in range(0, N):
        prod = np.dot(prod, A)
    return prod

def __min_walks(A, N, min_walks = None, min_dists = False):
    if N <= 0:
        return min_walks
    #min_walksN[i,j] = the minimum walk in N steps to get from i to j
    min_walks = A if min_walks is None else min_walks
    min_walks_update = min_walks.copy()
    for i in range(min_walks_update.shape[0]):
        for j in range(min_walks_update.shape[1]):
            #have: the minimum walk in N steps to get from i to anything
            #want: the minimum walk in N+1 steps to get from i to anything + from anything to j
            is_to_anythings = min_walks[i,:]
            anythings_to_j = A[:,j]
            min_walks_update[i, j] = (is_to_anythings + anythings_to_j).min()
            if min_dists:
                min_walks_update[i,j] = min(min_walks[i,j], min_walks_update[i,j])
    #print("min walks update: ", min_walks_update)
    return __min_walks(A, N-1, min_walks = min_walks_update, min_dists = min_dists)




def __PDP_factor(A):
    eigenvals, P = np.linalg.eigh(A)
    #eigenvals = eigenvals.astype(np.float64)
    #P = P.astype(np.float64)
    D = np.diag(eigenvals)
    return P, D

def __connected_component_cluster(A, thresh_dist):
    A = A.copy()
    #one traversal:
    #1) start at root
    #2) the children of a given node are all children whose p(n2 | n1) > thresh_dist
    clusters = []
    visited = np.zeros(A.shape[0], dtype = np.bool)
    while (visited != np.ones(A.shape[0], dtype = np.bool)).any():
        root = np.argmin(visited)
        q = Queue()
        visited[root] = True
        q.push(root)
        cluster = []
        while not q.isempty():
            n = q.pop()
            cluster.append(n)
            #traverse over connections FROM n TO other nodes, meaning along A[n]
            for transition in range(A.shape[0]):
                if A[n, transition] < thresh_dist and not visited[transition]:
                    q.push(transition)
                    visited[transition] = True
        clusters.append(cluster)
    clusters.sort(key = lambda cluster: len(cluster), reverse = True)
    return clusters


'''
def __normalize(A, target_sum = 1):
    if A.shape[0] == 1:
        return np.array([[target_sum]])
    first_col_sum = np.sum(A[:,0])
    col_vec = A[:,0].copy()
    A[:,0] = col_vec * target_sum/first_col_sum
    A[0,:] = col_vec * target_sum/first_col_sum

    fixed_sum = np.sum(A[:,0])
    assert abs(fixed_sum - target_sum) < 0.001, "fixed sum: " + str(fixed_sum) + ", target: " + str(target_sum)
    A[1:,1:] = __normalize(A.copy()[1:, 1:], target_sum = target_sum - A[1, 0])
    return A
'''
'''
def __normalize(A):
    for i in range(A.shape[0]):
        target_sum = 1-np.sum(A[:i, i])
        print("target sum: ", target_sum)
        if target_sum != 0:
            rest_sum = np.sum(A[i:,i])
            rest_col_vec = A[i:, i].copy()
            A[i:, i] = target_sum*rest_col_vec/rest_sum#target_sum * rest_col_vec/rest_sum
            A[i, i:] = target_sum*rest_col_vec/rest_sum#target_sum * rest_col_vec/rest_sum
    return A

def __normalize_softmax(A, target_sum = 1):
    if A.shape[0] == 1:
        return np.array([[target_sum]])
    col_vec = A[:, 0].copy()
    divisor = np.sum(np.exp(col_vec))
    numerator = target_sum * np.exp(col_vec)
    A[:,0] = numerator/divisor
    A[0,:] = numerator/divisor
    A[1:, 1:] = __normalize(A[1:, 1:].copy(), target_sum = target_sum - A[1,0])
    return A
'''

'''
    (1,2,3)/6 = 1
    (1,2,3)/12 = 6/12 = 1/2
    (1,2,3)/(2*6) = 1/2
    (1/2)*(1,2,3)/6 = 1/2
'''
