---
layout: post
title: Kaggle Spooky Author Identification - 0-29 Public LB score, Beginner NLP Tutorial
permalink : https://www.kaggle.com/skrcode/0-29-public-lb-score-beginner-nlp-tutorial
---

My published kernel in the Kaggle Contest, Spooky Author Identification. Highest Public LB score  - 0.29 for a published kernel.Uses Simple Feature Engineering like Punctuation,Stop Words,Glove Sentence vectors. In addition is creates stack features from simple Features such as tfidf and count vectors for words and chars and applying multinomial naive bayes(mnb)- tfidf+words+mnb,tfidf+chars+mnb,count+words+mnb,count+chars+mnb. Conv Nets on keras texttosequence, NNs on glove sentence vectors and Fast Text are also used as stack features. XGBoost, which is the final model which will use the simple features and stack features as input
