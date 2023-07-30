## <u> K-means Clustering </u>

K-means clustering is a type of unsupervised learning, which is used with unlabeled dataset. The goal of this algorithm is to find K groups in the data. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:

- The centroids of the K clusters, which can be used to label new data
- Labels for the training data (each data point is assigned to a single cluster)

K-means works by defining spherical clusters that are separable in a way so that the mean value converges towards the cluster center. Because of this, K-Means may underperform sometimes.

<b><u>Use Cases:</u></b>
- Document Classification
- Delivery Store Optimization
- Customer Segmentation
- Insurance Fraud Detection etc.

### <u> Algorithm </u>:

Κ-means clustering algorithm inputs are the number of clusters Κ and the data set. Algorithm starts with initial estimates for the Κ centroids, which can either be randomly generated or randomly selected from the data set. The algorithm then iterates between two steps:

<b><u>1. Data assigment step:</u></b>

Each centroid defines one of the clusters. In this step, each data point based on the squared Euclidean distance is assigned to its nearest centroid. If $c_i$ is the collection of centroids in set C, then each data point x is assigned to a cluster based on

$$\underset{c_i \in C}{\min} \; dist(c_i,x)^2$$

where dist( · ) is the standard (L2) Euclidean distance.

<b><u>2. Centroid update step:</u></b>

Centroids are recomputed by taking the mean of all data points assigned to that centroid's cluster.

The algorithm iterates between step one and two until a stopping criteria is met (no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).

<b>This algorithm may converge on a local optimum. </b> Assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.

<b><u>Choosing K</u></b>

If the true label is not known in advance, then K-Means clustering can be evaluated using <b> Elbow Criterion </b>, <b> Silhouette Coefficient </b> , cross-validation, information criteria, the information theoretic jump method, and the G-means algorithm. .


<b><u>Elbow Criterion Method:</u></b>

The idea behind elbow method is to run k-means clustering on a given dataset for a range of values of k (e.g k=1 to 10), for each value of k, calculate sum of squared errors (SSE).

Calculate the mean distance between data points and their cluster centroid. Increasing the number of clusters(K) will always reduce the distance to data points, thus decrease this metric, to the extreme of reaching zero when K is as same as the number of data points. So the goal is to choose a small value of k that still has a low SSE.

We run the algorithm for different values of K(say K = 10 to 1) and plot the K values against SSE(Sum of Squared Errors). And select the value of K for the elbow point.

<b><u>Silhouette Coefficient Method:</u></b>

A higher Silhouette Coefficient score relates to a model with better-defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores:

- The mean distance between a sample and all other points in the same class.
- The mean distance between a sample and all other points in the next nearest cluster.

The Silhouette Coefficient is for a single sample is then given as:

$$s=\frac{b-a}{max(a,b)}$$

To find the optimal value of k for KMeans, loop through 1..n for n_clusters in KMeans and calculate Silhouette Coefficient for each sample.

A higher Silhouette Coefficient indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

<u>K-Means algorithm uses Eucledean Distance, other popular distance metrics in Machine Learning are:</u>

1. <b> Cosine distance </b>: It determines the cosine of the angle between the point vectors of the two points in the n dimensional space. Closer the point vectors are by angle, the higher is the Cosine Similarity

$$\cos\theta = \frac{\overrightarrow{a} . \overrightarrow{b}}{\parallel \overrightarrow{a} \parallel \parallel \overrightarrow{b} \parallel} = \frac{\sum_{i=1}^{n} a_ib_i}{\sqrt{\sum_{i=1}^{n}a_i^2 \sqrt{\sum_{i=1}^{n}b_i^2}}}$$

where $\overrightarrow{a} . \overrightarrow{b} = \sum_{i=1}^{n}a_ib_i = a_1b_1 + a_2b_2 + ... + a_nb_n$

2. <b> Manhattan distance </b>: is the total sum of the difference between the x-coordinates  and y-coordinates.

$$ManhattanDistance = |x1 – x2| + |y1 – y2|$$

Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values. Various distance measures, or norms, are possible:

- Computing the root of a sum of squares (RMSE) corresponds to the Euclidian norm: it is the notion of distance you are familiar with. It is also called the ℓ2 norm(...)

- Computing the sum of absolutes (MAE) corresponds to the ℓ1 norm,(...). It is sometimes called the Manhattan norm because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.
