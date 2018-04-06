# Classification of Sentinel-2 Multispectral Images using Deep Neural Network in R-H2O (Windows 10)'


### Load R packages 
library(rgdal)  # spatial data processing
library(raster) # raster processing
library(plyr)   # data manipulation 
library(dplyr)  # data manipulation 
library(RStoolbox) # ploting spatial data 
library(ggplot2) # plotting 
library(RColorBrewer)
library(sp)

#### Set working directory
setwd("F:\\My_GitHub\\DNN_H20_R")

#### Load point and grid data
point<-read.csv("Data\\point_data.csv", header = T)
grid<-read.csv("Data\\grid_data.csv", header = T)

##### Creat a data frame
point.data<-cbind(point[c(4:13)],Class=point$Class)
grid.data<-grid[c(4:13)]
grid.xy<-grid[c(3,1:2)]

#### Install H2O
#install.packages("h20")

#### Start and Initialize  H20 local cluster

library(h2o)
localH2o <- h2o.init(nthreads = -1, max_mem_size = "50G") 

#### Import data to H2O cluster
df<-  as.h2o(point.data)
grid<- as.h2o(grid.data)

#### Split data into train, validation and test dataset
splits <- h2o.splitFrame(df, c(0.75,0.125), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 75%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 12%
test   <- h2o.assign(splits[[3]], "test.hex")  # 13%

#### Create response and features data sets
y <- "Class"
x <- setdiff(names(train), y)

### Deep Learning Model
dl_model <- h2o.deeplearning(
  model_id="Deep_Learning",                  # Destination id for this model
  training_frame=train,                      # Id of the training data frame
  validation_frame=valid,                    # Id of the validation data frame 
  x=x,                                       # a vector predictor variable
  y=y,                                       # name of reponse vaiables
  standardize=TRUE,                          # standardize the data
  score_training_samples=0,                  # training set samples for scoring (0 for all)
  activation = "RectifierWithDropout",       # Activation function
  score_each_iteration = TRUE,              
  hidden = c(200,200,200,200),               # 4 hidden layers, each of 200 neurons
  hidden_dropout_ratios=c(0.2,0.1,0.1,0),    # for improve generalization
  stopping_tolerance = 0.001,                # tolerance for metric-based stopping criterion
  epochs=100,                                # the dataset should be iterated (streamed)
  adaptive_rate=TRUE,                        # manually tuned learning rate
  l1=1e-6,                                   # L1/L2 regularization, improve generalization
  l2=1e-6,
  max_w2=10,                                 # helps stability for Rectifier
  nfolds=10,                                 # Number of folds for K-fold cross-validation
  fold_assignment="Stratified",              # Cross-validation fold assignment scheme
  keep_cross_validation_fold_assignment = TRUE,
  seed=125,
  reproducible = TRUE,
  variable_importances=T
) 


####  Model Summary
#summary(dl_model)
#capture.output(print(summary(dl_model)),file =  "DL_summary_model_01.txt")

#### Mean error
h2o.mean_per_class_error(dl_model, train = TRUE, valid = TRUE, xval = TRUE)

#### Scoring_history
scoring_history<-dl_model@model$scoring_history
#write.csv(scoring_history, "scoring_history_model_02.csv")

####  Plot the classification error over all epochs or samples.
plot(dl_model,
timestep = "epochs",
metric = "classification_error")

#### Plot logloss 
plot(dl_model,
  timestep = "epochs",
  metric = "logloss")

#### Plot RMSE
plot(dl_model,
  timestep = "epochs",
  metric = "rmse")

#### Cross-validation  Error
# Get the CV models from the deeplearning model object` object
cv_models <- sapply(dl_model@model$cross_validation_models, 
function(i) h2o.getModel(i$name))
# Plot the scoring history over time
  plot(cv_models[[1]], 
  timestep = "epochs", 
  metric = "classification_error")


####  Cross validation results
print(dl_model@model$cross_validation_metrics_summary%>%.[,c(1,2)])
#capture.output(print(dl_model@model$cross_validation_metrics_summary%>%.[,c(1,2)]),file =  "DL_CV_model_01.txt")

#### Model performance with Test data set
#### Compare the training error with the validation and test set errors

h2o.performance(dl_model, newdata=train)     ## full train data
h2o.performance(dl_model, newdata=valid)     ## full validation data
h2o.performance(dl_model, newdata=test)     ## full test data

#capture.output(print(h2o.performance(dl_model,test)),file =  "test_data_model_01.txt")


#### Confusion matrix
train.cf<-h2o.confusionMatrix(dl_model)
print(train.cf)
valid.cf<-h2o.confusionMatrix(dl_model,valid=TRUE)
print(valid.cf)
test.cf<-h2o.confusionMatrix(dl_model,test)
print(test.cf)
#write.csv(train.cf, "CFM_train_model_01.csv")
#write.csv(valid.cf, "CFM_valid_model_01.csv")
#write.csv(test.cf, "CFM_test_moldel_01.csv")


#### Grid Prediction
g.predict = as.data.frame(h2o.predict(object = dl_model, newdata = grid))

#### Stop h20 cluster
h2o.shutdown(prompt=FALSE)

#### Extract Prediction Class
grid.xy$Class<-g.predict$predict
str(grid.xy)
grid.xy.na<-na.omit(grid.xy)

#### Join Class Id Column
ID<-read.csv("Data\\Landuse_ID.csv", header=TRUE)
new.grid<-join(grid.xy.na, ID, by="Class", type="inner")
#write.csv(new.grid, "Predicted_Landuse_Class.csv")

#### Convert to raster and write
x<-SpatialPointsDataFrame(as.data.frame(new.grid)[, c("x", "y")], data = new.grid)
r <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Class_ID")])
#writeRaster(r,"predicted_Landuse.tiff","GTiff",overwrite=TRUE)

#### Plot and Save as a tiff file

myPalette <- colorRampPalette(c("khaki1","maroon1", "darkgreen","green", "blue"))
lu<-spplot(r,"Class_ID", main="Landuse Classes" , 
colorkey = list(space="right",tick.number=1,height=1, width=1.5,
labels = list(at = seq(1,4.8,length=5),cex=1.0,
lab = c("Road/parking/pavement" ,"Building", "Tree/buses", "Grass", "Water"))),
col.regions=myPalette,cut=4)
lu

windows(width=4, height=4)
  tiff( file="FIGURE_Landuse_Class.tif",
  width=4, 
  height=4,
  units = "in", 
  pointsize = 12, 
  res=600, 
  restoreConsole = T,
  compression =  "lzw",
  bg="transparent")
print(lu)
dev.off()
