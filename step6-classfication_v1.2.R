##
# Author Michael M. March 2019
# version 1.2.0
# NOTE THAT : library(e1071) SHOULD BE INSTALLED
# https://www.cs.umb.edu/~smimarog/textmining/datasets/
##

# load the data, either pca or pfa
TRAINING_ROWS = 6532#5485
loaded_file = file.choose()
data_loaded = read.csv(loaded_file, header = FALSE, ",")
print (paste ("Loaded corups :", loaded_file))
train_data = data_loaded[1:TRAINING_ROWS, ]
test_data = data_loaded[TRAINING_ROWS:nrow(data_loaded), ]
rm(data_loaded)
gc()

# print metadata
print (paste ("Training data size:", nrow(train_data), ncol(train_data)))
print (paste ("Testing data size: ", nrow(test_data), ncol (train_data)))
print (paste ("TOTAL SIZE OF CORUPS :", nrow(train_data) + nrow(test_data)))

# train loaded data using SVM
library(e1071)
svm_model = svm(V1 ~ ., data = train_data, kernel = "linear", cost = 10, scale = FALSE) # train svm model
prediction = predict(svm_model, test_data, type = "class") # predict the test data
confusion_matrix = table(predict = prediction, truth = test_data$V1) # construct confusion matrix
print(confusion_matrix)
recognition_rate = (sum(diag(confusion_matrix)) / sum(confusion_matrix)) * 100 # calculate recognition rate

# write to file
print (paste ("recognition rate :", recognition_rate))
write.csv (confusion_matrix, "confusion_matrix_PCA.csv")
#END#