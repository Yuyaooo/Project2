ST558Project2
================
Yuyao Liu
10/14/2020

# Introduction

I will analyze the Online News Popularity Data. This dataset summarizes
a heterogeneous set of features about articles published by Mashable in
a period of two years. There are 61 variables in the dataset which
contains 58 predictive attributes, 2 non-predictive attributes and one
target variable ‘share’. I choose some of the predictive variables that
can contains most of the information. For example, I choose some
variables of average values instead of minimum or maximum, so it may get
rid of some outliers. Also, some variables are about sentiment polarity
or subjectivity, I choose one variable of these two types that can
represent most information about word tokens. Meanwhile, I deleted some
variables that are highly correlated with some others.

## Predictors:

1.  n\_tokens\_title: Number of words in the title

2.  n\_tokens\_content: Number of words in the content

3.  n\_non\_stop\_unique\_tokens: Rate of unique non-stop words in the
    content

4.  num\_hrefs: Number of links

5.  num\_self\_hrefs: Number of links to other articles published by
    Mashable

6.  num\_imgs: Number of images

7.  num\_videos: Number of videos

8.  average\_token\_length: Average length of the words in the content

9.  num\_keywords: Number of keywords in the metadata

10. data\_channel\_is\_lifestyle: Is data channel ‘Lifestyle’?

11. data\_channel\_is\_entertainment: Is data channel ‘Entertainment’?

12. data\_channel\_is\_bus: Is data channel ‘Business’?

13. data\_channel\_is\_socmed: Is data channel ‘Social Media’?

14. data\_channel\_is\_tech: Is data channel ‘Tech’?

15. data\_channel\_is\_world: Is data channel ‘World’?

16. kw\_avg\_min: Worst keyword (avg. shares)

17. kw\_avg\_max: Best keyword (avg. shares)

18. kw\_avg\_avg: Avg. keyword (avg. shares)

19. self\_reference\_avg\_sharess: Avg. shares of referenced articles in
    Mashable

20. LDA\_00: Closeness to LDA topic 0

21. LDA\_01: Closeness to LDA topic 1

22. LDA\_02: Closeness to LDA topic 2

23. LDA\_03: Closeness to LDA topic 3

24. LDA\_04: Closeness to LDA topic 4

25. global\_rate\_positive\_words: Rate of positive words in the content

26. global\_rate\_negative\_words: Rate of negative words in the content

27. avg\_positive\_polarity: Avg. polarity of positive words

28. avg\_negative\_polarity: Avg. polarity of negative words

29. title\_subjectivity: Title subjectivity

30. title\_sentiment\_polarity: Title polarity

## Response/Target

shares: Number of shares

## Goal

The goal is to predict the number of shares(`shares`) in social
networks. I will use regression tree and boosted tree to predict shares,
and compare them using test set.

``` r
library(rmarkdown)
library(tidyverse)
library(caret)
library(corrplot)
```

# Data

``` r
path <- file.path(getwd(), "OnlineNewsPopularity.csv")
temp_news <- read_csv(file = path)
weekday <- vector()
for(i in seq_len(nrow(temp_news))){
  if (temp_news$weekday_is_monday[i] == 1){
    weekday[i]  <- "monday"
  }
  else if (temp_news$weekday_is_tuesday[i] == 1){
    weekday[i]  <- "tuesday"
  }
  else if (temp_news$weekday_is_wednesday[i] == 1){
    weekday[i]  <- "wednesday"
  }
  else if (temp_news$weekday_is_thursday[i] == 1){
    weekday[i]  <- "thursday"
  }
  else if (temp_news$weekday_is_friday[i] == 1){
    weekday[i]  <- "friday"
  }
  else if (temp_news$weekday_is_saturday[i] == 1){
    weekday[i]  <- "saturday"
  }
  else if(temp_news$weekday_is_sunday[i] == 1){
    weekday[i]  <- "sunday"
  }
}
temp_news <- cbind(temp_news, weekday)
news <- temp_news %>% filter(weekday == params$weekday) %>% select(n_tokens_title, n_tokens_content, n_non_stop_unique_tokens, num_hrefs, num_self_hrefs, num_imgs, num_videos, average_token_length, num_keywords, data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed,data_channel_is_tech,data_channel_is_world,kw_avg_min, kw_avg_max,kw_avg_avg,self_reference_avg_sharess,LDA_00,LDA_01,LDA_02, LDA_03, LDA_04, global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,title_subjectivity,title_sentiment_polarity,shares)
news$data_channel_is_lifestyle <- as.factor(news$data_channel_is_lifestyle)
news$data_channel_is_entertainment <- as.factor(news$data_channel_is_entertainment)
news$data_channel_is_bus <- as.factor(news$data_channel_is_bus)
news$data_channel_is_socmed <- as.factor(news$data_channel_is_socmed)
news$data_channel_is_tech <- as.factor(news$data_channel_is_tech)
news$data_channel_is_world <- as.factor(news$data_channel_is_world)
set.seed(558)
train <- sample(1:nrow(news), size = nrow(news)*0.7) 
test <- dplyr::setdiff(1:nrow(news), train) 
newsTrain <- news[train, ]
newsTest <- news[test, ]
```

# Summarizations

## Summary

The summary of training data includes minimum, 1st quantile, median,
mean, 3rd quantile and maximum.

``` r
sum_news <- select(newsTrain, -contains("data_channel"))
summary(sum_news)
```

    ##  n_tokens_title  n_tokens_content n_non_stop_unique_tokens
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :0.0000          
    ##  1st Qu.: 9.00   1st Qu.: 248.0   1st Qu.:0.6289          
    ##  Median :10.00   Median : 402.0   Median :0.6924          
    ##  Mean   :10.43   Mean   : 542.8   Mean   :0.6747          
    ##  3rd Qu.:12.00   3rd Qu.: 719.8   3rd Qu.:0.7553          
    ##  Max.   :18.00   Max.   :7764.0   Max.   :1.0000          
    ##    num_hrefs      num_self_hrefs      num_imgs        num_videos    
    ##  Min.   :  0.00   Min.   : 0.000   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.: 1.000   1st Qu.: 0.000  
    ##  Median :  7.00   Median : 3.000   Median : 1.000   Median : 0.000  
    ##  Mean   : 10.69   Mean   : 3.362   Mean   : 4.414   Mean   : 1.326  
    ##  3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.: 3.000   3rd Qu.: 1.000  
    ##  Max.   :162.00   Max.   :51.000   Max.   :93.000   Max.   :75.000  
    ##  average_token_length  num_keywords     kw_avg_min        kw_avg_max    
    ##  Min.   :0.000        Min.   : 1.00   Min.   :   -1.0   Min.   :     0  
    ##  1st Qu.:4.482        1st Qu.: 6.00   1st Qu.:  133.2   1st Qu.:174200  
    ##  Median :4.655        Median : 7.00   Median :  227.4   Median :244633  
    ##  Mean   :4.549        Mean   : 7.14   Mean   :  309.6   Mean   :259644  
    ##  3rd Qu.:4.842        3rd Qu.: 9.00   3rd Qu.:  349.4   3rd Qu.:332793  
    ##  Max.   :8.042        Max.   :10.00   Max.   :29946.9   Max.   :843300  
    ##    kw_avg_avg    self_reference_avg_sharess     LDA_00       
    ##  Min.   :    0   Min.   :     0             Min.   :0.01818  
    ##  1st Qu.: 2367   1st Qu.:  1024             1st Qu.:0.02525  
    ##  Median : 2841   Median :  2200             Median :0.03351  
    ##  Mean   : 3064   Mean   :  6054             Mean   :0.18917  
    ##  3rd Qu.: 3539   3rd Qu.:  5200             3rd Qu.:0.25143  
    ##  Max.   :33536   Max.   :423100             Max.   :0.91999  
    ##      LDA_01            LDA_02            LDA_03            LDA_04       
    ##  Min.   :0.01819   Min.   :0.01819   Min.   :0.01819   Min.   :0.01818  
    ##  1st Qu.:0.02504   1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02857  
    ##  Median :0.03337   Median :0.04000   Median :0.04000   Median :0.04001  
    ##  Mean   :0.15282   Mean   :0.20568   Mean   :0.21825   Mean   :0.23408  
    ##  3rd Qu.:0.17169   3rd Qu.:0.31347   3rd Qu.:0.35395   3rd Qu.:0.41033  
    ##  Max.   :0.91997   Max.   :0.92000   Max.   :0.91998   Max.   :0.92707  
    ##  global_rate_positive_words global_rate_negative_words
    ##  Min.   :0.00000            Min.   :0.00000           
    ##  1st Qu.:0.02840            1st Qu.:0.00978           
    ##  Median :0.03828            Median :0.01534           
    ##  Mean   :0.03923            Mean   :0.01672           
    ##  3rd Qu.:0.04962            3rd Qu.:0.02169           
    ##  Max.   :0.12500            Max.   :0.09058           
    ##  avg_positive_polarity avg_negative_polarity title_subjectivity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :0.0000    
    ##  1st Qu.:0.3049        1st Qu.:-0.3280       1st Qu.:0.0000    
    ##  Median :0.3579        Median :-0.2532       Median :0.1000    
    ##  Mean   :0.3543        Mean   :-0.2587       Mean   :0.2733    
    ##  3rd Qu.:0.4117        3rd Qu.:-0.1854       3rd Qu.:0.5000    
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   :1.0000    
    ##  title_sentiment_polarity     shares      
    ##  Min.   :-1.00000         Min.   :     1  
    ##  1st Qu.: 0.00000         1st Qu.:   914  
    ##  Median : 0.00000         Median :  1400  
    ##  Mean   : 0.06453         Mean   :  3639  
    ##  3rd Qu.: 0.13636         3rd Qu.:  2775  
    ##  Max.   : 1.00000         Max.   :690400

## Correlation

We can explore the data using correlation especially how these variables
as predictors are correlated with our target response `shares`.

Due to the large amount of variables, we may visualize them as groups.
The first nine variables are about numbers. Let’s see the correlation of
first nine variables and `shares`.

``` r
corr1 <- select(sum_news, 1:9, shares) %>% cor()
corrplot(corr1)
```

![](mondayAnalysis_files/figure-gfm/unnamed-chunk-110-1.png)<!-- -->

Most of variables are not highly correlated with others.

Let’s see the variables about the content and word use and response
`shares`.

``` r
corr2 <- select(sum_news, 10:18, shares) %>% cor()
corrplot(corr2)
```

![](mondayAnalysis_files/figure-gfm/unnamed-chunk-111-1.png)<!-- -->

``` r
corr3 <- select(sum_news, -c(1:18)) %>% cor()
corrplot(corr3)
```

![](mondayAnalysis_files/figure-gfm/unnamed-chunk-111-2.png)<!-- -->

**All the predictors are little correlated to `shares`. **

# Modeling

## Nonlinear model

### Regression tree

The first model I create is a tree-based model chosen using leave one
out cross validation. The response for this data is continuous, so I use
regression tree model.

**How do we fit the regression tree model?**

We fit the model using greedy algorithm. For every possible value of
each predictor, find residual sum of squares and minimize them.

I standardize the numeric predictors by centering and scaling.
Meanwhile, I determine tuning parameter choises using leave one out
cross validation.

The final chosen model is:

``` r
set.seed(558)
regTree <- train(shares ~ ., data = newsTrain, method = "rpart",
trControl = trainControl(method = "LOOCV"), preProcess = c("center", "scale"))
regTree
```

    ## CART 
    ## 
    ## 4662 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 4661, 4661, 4661, 4661, 4661, 4661, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           RMSE      Rsquared      MAE     
    ##   0.004556157  14372.11  0.0011746332  3742.516
    ##   0.006043462  14393.03  0.0006597003  3827.473
    ##   0.035987524  14333.77  0.0005650235  4209.543
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.03598752.

``` r
regTree$bestTune
```

### Boosted tree - Ensemble

The second model I create is a boosted tree model chosen using
cross-validation.

**How do we fit the boosting model?**

The trees grown sequentially. Each subsequent tree is grown on a
modified version of original data. Predictions updated as trees grown.

Procedure:

1.  Initialize predictions as 0

2.  Find the residuals (observed-predicted), call the set of them *r*

3.  Fit a tree with *d* splits (*d*+1 terminal nodes) tresting the
    residuals as the response (which they are for the first fit)

4.  Update predictions

5.  Update residuals for new predictions and repeat *B* times

I also standardize the numeric predictors by centering and scaling and
determine tuning parameter choises using cross-validation.

The final chosen model is:

``` r
set.seed(558)
boostTree <- train(shares ~ ., data = newsTrain, method = "gbm",
trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), verbose = FALSE)
boostTree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4662 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4196, 4195, 4195, 4195, 4197, 4197, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   50      11796.67  0.015234181  3636.251
    ##   1                  100      11836.38  0.015354384  3638.325
    ##   1                  150      11932.82  0.014717222  3643.386
    ##   2                   50      12008.38  0.008782419  3650.524
    ##   2                  100      12159.21  0.009766898  3713.406
    ##   2                  150      12430.49  0.007372620  3756.155
    ##   3                   50      11985.98  0.013457498  3689.072
    ##   3                  100      12116.95  0.013329811  3727.563
    ##   3                  150      12312.23  0.008529315  3815.454
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50,
    ##  interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boostTree$bestTune
```

### Test and Compare

``` r
tree_pred <- postResample(predict(regTree, newdata = newsTest), newsTest$shares)
boost_pred <- postResample(predict(boostTree, newdata = newsTest), newsTest$shares)
round(rbind(tree_pred, boost_pred), 4)
```

    ##                RMSE Rsquared      MAE
    ## tree_pred  16484.28       NA 3687.826
    ## boost_pred 16411.44   0.0101 3643.421

Choose the model with smaller root mean square error (RMSE), smaller
mean absolute error (MAE) and bigger R squared.
