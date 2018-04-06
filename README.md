
# Deep Neural Network with H2O-R: Satellite-Image Classification

This tutorial will show how to implement [Deep Neural Network](https://en.wikipedia.org/wiki/Deep_learning) for pixel based [supervised classification](https://gis.stackexchange.com/questions/237461/distinction-between-pixel-based-and-object-based-classification) of [Sentinel-2 multispectral images](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) using [H20](http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html) package in [R](https://cloud.r-project.org/). 

[H2O is an open source, in-memory, distributed, fast, and scalable machine learning and predictive analytics platform that allows you to build machine learning models on big data and provides easy productionalization of those models in an enterprise environment](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html). Itâs core code is written in Java and can read data in parallel from a distributed cluster and also from local culster. H2Oâs REST API allows access to all the capabilities of H2O from an external program or script via JSON over HTTP. The Rest API is used by H2Oâs [web interface (Flow UI)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html), [R binding (H2O-R)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html), and [Python binding (H2O-Python)](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html). Requirement and installation steps in diffident platforms such as R and Python are available here.

To classify pixel-based images, We will use the Deep Neural Network algorithm for pixel-based image classification using [H20](http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html) package within the [R](https://www.r-project.org/)].We will use use two data set- "point_data" and "grid_data". We will will split "point_data" into a training set (75% of the data), a validation set (12%) and a test set (13%) data.The validation data set will be used to optimize the model parameters during training process.The model's performance will be tested with an independent test data set.  

**Tuning and Optimizations parameters:** 

* We will use four hidden layers with 200 neurons and Rectifier Linear (ReLU) as a activation function of neurons. 
* The default stochastic gradient descent function will be used to optimize different objective functions and to minimize training loss. 
* To reduce the generalization error and the risk of over-fitting of the model, we will use set low values for L1  and L2 regularizations.
* The model will be cross validated with 10 folds with stratified sampling 
* The model will be run with 100 epochs. 

More details of Tuning and Optimizations parameters of H20 Deep Neural Network for supervised classification can be found [here](http://docs.h2o.ai/h2o-tutorials/latest-stable/tutorials/deeplearning/index.html)
