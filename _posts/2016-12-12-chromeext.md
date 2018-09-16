---
layout: post
title: Quora Chrome Extension
permalink : https://github.com/skrcode/Chrome-Extension
categories: [projects]
---

Hobby Project - 
Quora Chrome Extension which classifies Quora answers into genres such as information,stories and world affairs for improved reading experience

The test data included contains data from the facebook page "Best-Of-Quora"

Gets labelled training data from python nltk's "Brown dataset". 
Uses Google's Word2Vec to perform meaning analysis and get data points for each word. 
Uses K means clustering to create an unsupervised model of for the list of words.
Associates each word to a cluster(centroid) and creates a bag of cluster model.
Using this bag of cluster vector for each paragraph, a RandomForest model with suitable parameter tuning is trained.
The test data is run on this model.

The Chrome Extension fetches data from the Quora home page and hits the python service and retrieves the output