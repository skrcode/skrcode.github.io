---
layout: post
title: Kaggle Spooky Author Identification - 0.29, Highest Public LB score for a published kernel
permalink : https://www.kaggle.com/skrcode/0-29-public-lb-score-beginner-nlp-tutorial
categories: [blogs,projects]
---

My published kernel in the Kaggle Contest, Spooky Author Identification. Highest Public LB score - 0.29 (Top 50) for a published kernel in the contest. Uses Simple Feature Engineering like Punctuation,Stop Words,Glove Sentence vectors. 
In addition, it creates stack features from simple Features such as tfidf and count vectors for words and chars. Multinomial naive bayes(mnb) is then applied with the following combination - tfidf+words+mnb,tfidf+chars+mnb,count+words+mnb,count+chars+mnb. 
Conv Nets on keras texttosequence, NNs on glove sentence vectors and Fast Text are also used as stack features. 
XGBoost, which is the final model which will use the simple and stack features as input.
