#  Deep Neural Network with H20-R: Satellite-Image Classification

This tutorial will show how to implement [Deep Neural Network](https://en.wikipedia.org/wiki/Deep_learning) for pixel based [supervised classification](https://gis.stackexchange.com/questions/237461/distinction-between-pixel-based-and-object-based-classification) of [Sentinel-2 multispectral images](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) using [H20](http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html) package in [R](https://cloud.r-project.org/). 

[H2O is an open source, in-memory, distributed, fast, and scalable machine learning and predictive analytics platform that allows you to build machine learning models on big data and provides easy productionalization of those models in an enterprise environment](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html). It's core code is written in Java and can read data in parallel from a distributed cluster and also from local culster. H2O allows access to all the capabilities of H2O from an external program or script via JSON over HTTP. The Rest API is used by H2O's [web interface (Flow UI)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html), [R binding (H2O-R)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html), and [Python binding (H2O-Python)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html). Requirement and installation steps in  R can be found here [here](http://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/2/index.html).

We will use the Deep Neural Network algorithm using [H20](http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html) package in  [R](https://www.r-project.org/) for image classification. First, we will split "point_data" into a training set (75% of the data), a validation set (12%) and a test set (13%) data.The validation data set will be used to optimize the model parameters during training process.The model's performance will be tested with the data set and then we will predict landuse clasess on grid data set. The point and grid data can be download as [rar](https://www.dropbox.com/s/l94zhzwjrc3lkk7/Point_Grid_Data.rar?dl=0), [7z](https://www.dropbox.com/s/77qk7raj48z0151/Point_Grid_Data.7z?dl=0) and [zip](https://www.dropbox.com/s/007vd9vayn60c2s/Point_Grid_Data.zip?dl=0) format. 

**Tuning and Optimizations parameters:** 

* Four hidden layers with 200 neurons and Rectifier Linear (ReLU) as a activation function of neurons. 
* The default stochastic gradient descent function will be used to optimize different objective functions and to minimize training loss. 
* To reduce the generalization error and the risk of over-fitting of the model, we will use set low values for L1  and L2 regularizations.
* The model will be cross validated with 10 folds with stratified sampling 
* The model will be run with 100 epochs. 

More details of Tuning and Optimizations parameters of H20 Deep Neural Network for supervised classification can be found [here](http://docs.h2o.ai/h2o-tutorials/latest-stable/tutorials/deeplearning/index.html)
