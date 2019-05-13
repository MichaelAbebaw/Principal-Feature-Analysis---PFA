##
# Author Michael M. March 2019
# version 1.2.0
# https://www.cs.umb.edu/~smimarog/textmining/datasets/
##

# load the data, i.e. formatted as term-by-document
loaded_file = file.choose()
data_loaded = read.csv(loaded_file, TRUE, ",")
print(loaded_file)

# strip the terms column
print (paste ("Operation Started : ", Sys.time()))
data_striped = data_loaded[-1]
rm(data_loaded) # remove the loaded data to save memory

# transpose the data frame to change it to document-by-term matrix
print (paste ("Transposing data started : ", Sys.time()))
data_transposed = t(data_striped)
rm(data_striped)

# run pca to know which columns to keep on col_to_keep
print (paste ("PCA calculation started : ", Sys.time()))
pca = prcomp(data_transposed, center = TRUE, scale. = TRUE)
vars = apply(pca$x, 2, var)
rm(pca)
to_keep = cumsum(vars / sum(vars))
pc = data.frame(to_keep)
cols_to_keep = data.frame (which(pc$to_keep == 1))[1,]
print(paste ("Columns to keep :", cols_to_keep))
gc()

# get the covariance matrix of the transposed frame (document-by-term)
print (paste ("Evaluating covariance matrix started : ", Sys.time()))
covariance_matrix = cov(data_transposed)

# get the eigen vector and value for the covariance matrix
print (paste ("Evaluating the eigen vectors started : ", Sys.time()))
e = eigen (covariance_matrix)
rm (covariance_matrix)
eigen_value = e$values
eigen_vector = e$vectors
rm(e)
gc()

# calculate the retained variability
retained_variability = (sum(eigen_value[1:cols_to_keep]) / sum(eigen_value)) * 100
rm(eigen_value)
print(paste ("Retained variability :", retained_variability))

####### PCA ##############
print (paste ("PCA calculation started : ", Sys.time()))
pca_transformed_data = data_transposed %*% eigen_vector[, 1:cols_to_keep] # use dot product to transform the data into new space
write.csv(pca_transformed_data, "r52-corpus_train_test_PCA.csv")
rm(pca_transformed_data)
##########################

####### PFA ##############
print (paste ("K-Means clustering started : ", Sys.time()))
data_points = eigen_vector[, 1:cols_to_keep] # eigen vectors to data points
rm(eigen_vector)
gc()
cluster_center = cols_to_keep + 5 # at least greater than the columns to keep
data_clusters = kmeans(data_points, centers = cluster_center, iter.max = 99999999) # kmeans clusters
data_cluster_centers = data_clusters$centers
plot(data_cluster_centers)
rm(data_clusters)

# find the data points that are near to the cluster centers
print (paste ("Generating PFA index started : ", Sys.time()))
pfa_term_index = vector("list", nrow (data_cluster_centers)) # create a list that corrospond to the number of clusters
rows = nrow (data_cluster_centers)
for (i in 1:rows) {
  # for each cluster
  cluster_center = data_cluster_centers[i, ] # find point of the cluster
  minimum_distance = 9999999
  pfa_index = -1
  for (j in 1:nrow(data_points)) {
    # for each data points in the eigen vector find the point near to the cluster point
    distance = dist(rbind (cluster_center, data_points[j, ]))
    if (minimum_distance > distance) {
      minimum_distance = distance
      pfa_index = j
    }
  }
  pfa_term_index[[i]] = pfa_index # save the index of the rows of the data points
  print(paste (round((i / rows)*100, 2) , "%"))
}
rm(data_cluster_centers)
rm(data_points)
rm(cluster_center)
gc()

# pick the columns from the transposed data
index_to_pick = sort(unlist(pfa_term_index, use.names = FALSE))
pfa_transformed_data = data_transposed[, c(index_to_pick)]
write.csv(pfa_transformed_data, "r52-corpus_train_test_PFA.csv")
rm(list=ls())
gc ()