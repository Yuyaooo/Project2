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
library(GGally)
```

# Data

``` r
temp_news <- read_csv(file = "./OnlineNewsPopularity.csv")
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
np <- cbind(temp_news, weekday)
news <- np %>% filter(weekday == params$weekday) %>% select(n_tokens_title, n_tokens_content, n_non_stop_unique_tokens, num_hrefs, num_self_hrefs, num_imgs, num_videos, average_token_length, num_keywords, data_channel_is_lifestyle, data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed,data_channel_is_tech,data_channel_is_world,kw_avg_min, kw_avg_max,kw_avg_avg,self_reference_avg_sharess,LDA_00,LDA_01,LDA_02, LDA_03, LDA_04, global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,title_subjectivity,title_sentiment_polarity,shares)
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
mean, 3rd quantile and maximum values of each variable.

``` r
sum_news <- select(newsTrain, -contains("data_channel"))
summary(sum_news)
```

    ##  n_tokens_title  n_tokens_content n_non_stop_unique_tokens   num_hrefs     
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :  0.0000         Min.   :  0.00  
    ##  1st Qu.: 9.00   1st Qu.: 248.0   1st Qu.:  0.6309         1st Qu.:  4.00  
    ##  Median :10.00   Median : 402.0   Median :  0.6907         Median :  7.00  
    ##  Mean   :10.44   Mean   : 544.6   Mean   :  0.7994         Mean   : 10.74  
    ##  3rd Qu.:12.00   3rd Qu.: 690.0   3rd Qu.:  0.7548         3rd Qu.: 13.00  
    ##  Max.   :19.00   Max.   :7081.0   Max.   :650.0000         Max.   :304.00  
    ##  num_self_hrefs      num_imgs         num_videos    average_token_length
    ##  Min.   : 0.000   Min.   :  0.000   Min.   : 0.00   Min.   :0.000       
    ##  1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.00   1st Qu.:4.479       
    ##  Median : 3.000   Median :  1.000   Median : 0.00   Median :4.663       
    ##  Mean   : 3.352   Mean   :  4.548   Mean   : 1.35   Mean   :4.550       
    ##  3rd Qu.: 4.000   3rd Qu.:  4.000   3rd Qu.: 1.00   3rd Qu.:4.849       
    ##  Max.   :54.000   Max.   :100.000   Max.   :73.00   Max.   :7.975       
    ##   num_keywords      kw_avg_min        kw_avg_max       kw_avg_avg     
    ##  Min.   : 1.000   Min.   :   -1.0   Min.   :  3460   Min.   :  728.2  
    ##  1st Qu.: 6.000   1st Qu.:  140.7   1st Qu.:173040   1st Qu.: 2372.3  
    ##  Median : 7.000   Median :  235.9   Median :244133   Median : 2845.1  
    ##  Mean   : 7.165   Mean   :  306.1   Mean   :262756   Mean   : 3125.3  
    ##  3rd Qu.: 9.000   3rd Qu.:  359.4   3rd Qu.:334800   3rd Qu.: 3559.9  
    ##  Max.   :10.000   Max.   :15851.2   Max.   :843300   Max.   :27391.6  
    ##  self_reference_avg_sharess     LDA_00            LDA_01            LDA_02       
    ##  Min.   :     0             Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
    ##  1st Qu.:  1000             1st Qu.:0.02506   1st Qu.:0.02501   1st Qu.:0.02857  
    ##  Median :  2267             Median :0.03337   Median :0.03334   Median :0.04003  
    ##  Mean   :  6394             Mean   :0.18175   Mean   :0.13465   Mean   :0.21720  
    ##  3rd Qu.:  5300             3rd Qu.:0.23904   3rd Qu.:0.12950   3rd Qu.:0.33159  
    ##  Max.   :690400             Max.   :0.91998   Max.   :0.91994   Max.   :0.92000  
    ##      LDA_03            LDA_04        global_rate_positive_words
    ##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000           
    ##  1st Qu.:0.02857   1st Qu.:0.02858   1st Qu.:0.02857           
    ##  Median :0.04000   Median :0.05000   Median :0.03902           
    ##  Mean   :0.22061   Mean   :0.24561   Mean   :0.03962           
    ##  3rd Qu.:0.35567   3rd Qu.:0.43956   3rd Qu.:0.05000           
    ##  Max.   :0.91989   Max.   :0.92000   Max.   :0.11458           
    ##  global_rate_negative_words avg_positive_polarity avg_negative_polarity
    ##  Min.   :0.000000           Min.   :0.0000        Min.   :-1.0000      
    ##  1st Qu.:0.009479           1st Qu.:0.3042        1st Qu.:-0.3258      
    ##  Median :0.015267           Median :0.3563        Median :-0.2502      
    ##  Mean   :0.016494           Mean   :0.3510        Mean   :-0.2575      
    ##  3rd Qu.:0.021459           3rd Qu.:0.4076        3rd Qu.:-0.1833      
    ##  Max.   :0.135294           Max.   :0.8000        Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity     shares      
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :    42  
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:   896  
    ##  Median :0.1000     Median : 0.00000         Median :  1300  
    ##  Mean   :0.2779     Mean   : 0.07446         Mean   :  3163  
    ##  3rd Qu.:0.5000     3rd Qu.: 0.13636         3rd Qu.:  2500  
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :441000

## Correlation

We can explore the data using correlation especially how these variables
as predictors are correlated with our target response `shares`.

Due to the large amount of variables, we can visualize them as groups. I
divide 30 predictors into 4 groups. Each group has 6 predictors and one
response `share`.

*group1*

``` r
subnews1 <- select(sum_news, 1:6, shares)
corr1 <- cor(subnews1)
corrplot(corr1)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

*group 2*

``` r
subnews2 <- select(sum_news, 7:12, shares)
corr2 <- cor(subnews2)
corrplot(corr2)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

*group3*

``` r
subnews3 <- select(sum_news, 13:18, shares)
corr3 <- cor(subnews3)
corrplot(corr3)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

*group4*

``` r
subnews4 <- select(sum_news, -c(1:18))
corr4 <- cor(subnews4)
corrplot(corr4)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

The light color and smaller size of circle mean that the absolute
correlation value is small. Thus, the graphs above show that most of
predictors are not highly correlated with others. Meanwhile, **all the
predictors are little correlated to `shares`, even the correlations
between predictors and response are almost equal to 0.**

## Plots of relationships between predictors and response

*group1*

``` r
ggpairs(subnews1)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

*group2*

``` r
ggpairs(subnews2)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

*group3*

``` r
ggpairs(subnews3)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

*group4*

``` r
ggpairs(subnews4)
```

![](tuesdayAnalysis_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

The pairs of plots show the relationships between each two variables.
The last row of plots show the relationship between predictors and
`shares`.

**As the pairs plots shown, `shares` is less related to the
predictors.**

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
    ## 5173 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 5172, 5172, 5172, 5172, 5172, 5172, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           RMSE      Rsquared      MAE     
    ##   0.007187079  10952.27  0.0010228359  3171.830
    ##   0.013207905  10970.36  0.0004473620  3297.843
    ##   0.015804044  10468.00  0.0008819138  3559.241
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.01580404.

**The final chosen model with best tuning parameters**

``` r
regTree$bestTune %>% knitr::kable()
```

|   |       cp |
| - | -------: |
| 3 | 0.015804 |

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
    ## 5173 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4657, 4656, 4656, 4655, 4655, 4657, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   50      8957.586  0.025421011  2901.159
    ##   1                  100      8964.567  0.026857616  2886.596
    ##   1                  150      8972.347  0.026409831  2890.381
    ##   2                   50      9070.077  0.020514356  2939.540
    ##   2                  100      9184.411  0.016857136  2999.669
    ##   2                  150      9298.322  0.015296100  3057.889
    ##   3                   50      9203.092  0.012329159  2988.125
    ##   3                  100      9361.043  0.010141818  3076.241
    ##   3                  150      9478.636  0.008408282  3163.746
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning
    ##  parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

**The final chosen model with best tuning parameters**

``` r
boostTree$bestTune %>% knitr::kable()
```

| n.trees | interaction.depth | shrinkage | n.minobsinnode |
| ------: | ----------------: | --------: | -------------: |
|      50 |                 1 |       0.1 |             10 |

### Test and Compare

``` r
tree_pred <- postResample(predict(regTree, newdata = newsTest), newsTest$shares)
boost_pred <- postResample(predict(boostTree, newdata = newsTest), newsTest$shares)
round(rbind(tree_pred, boost_pred), 4)
```

    ##                RMSE Rsquared      MAE
    ## tree_pred  9033.748       NA 3142.650
    ## boost_pred 8840.989   0.0434 3018.345

Smaller root mean square error (RMSE) and smaller mean absolute error
(MAE) shows that we have a better fit of data and more accurate
predictions in test dataset.

Choose the better model with relatively smaller root mean square error
(RMSE), smaller mean absolute error (MAE) and bigger R squared.
