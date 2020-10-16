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
    ##  Min.   : 3.00   Min.   :   0.0   Min.   :0.0000           Min.   :  0.00  
    ##  1st Qu.: 9.00   1st Qu.: 239.0   1st Qu.:0.6306           1st Qu.:  4.00  
    ##  Median :10.00   Median : 397.5   Median :0.6977           Median :  7.00  
    ##  Mean   :10.38   Mean   : 523.6   Mean   :0.6774           Mean   : 10.66  
    ##  3rd Qu.:12.00   3rd Qu.: 680.0   3rd Qu.:0.7603           3rd Qu.: 13.00  
    ##  Max.   :23.00   Max.   :7413.0   Max.   :1.0000           Max.   :186.00  
    ##  num_self_hrefs      num_imgs         num_videos     average_token_length
    ##  Min.   : 0.000   Min.   :  0.000   Min.   : 0.000   Min.   :0.000       
    ##  1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.000   1st Qu.:4.467       
    ##  Median : 2.000   Median :  1.000   Median : 0.000   Median :4.656       
    ##  Mean   : 3.051   Mean   :  4.356   Mean   : 1.261   Mean   :4.537       
    ##  3rd Qu.: 4.000   3rd Qu.:  3.000   3rd Qu.: 1.000   3rd Qu.:4.855       
    ##  Max.   :51.000   Max.   :108.000   Max.   :91.000   Max.   :6.486       
    ##   num_keywords      kw_avg_min        kw_avg_max       kw_avg_avg     
    ##  Min.   : 1.000   Min.   :   -1.0   Min.   :  5362   Min.   :  776.1  
    ##  1st Qu.: 6.000   1st Qu.:  142.0   1st Qu.:173470   1st Qu.: 2379.2  
    ##  Median : 7.000   Median :  232.3   Median :245930   Median : 2857.4  
    ##  Mean   : 7.212   Mean   :  302.8   Mean   :260323   Mean   : 3142.7  
    ##  3rd Qu.: 9.000   3rd Qu.:  353.9   3rd Qu.:331819   3rd Qu.: 3603.6  
    ##  Max.   :10.000   Max.   :34855.1   Max.   :843300   Max.   :37607.5  
    ##  self_reference_avg_sharess     LDA_00            LDA_01            LDA_02       
    ##  Min.   :     0.0           Min.   :0.01818   Min.   :0.01818   Min.   :0.01818  
    ##  1st Qu.:   980.8           1st Qu.:0.02506   1st Qu.:0.02501   1st Qu.:0.02857  
    ##  Median :  2250.0           Median :0.03337   Median :0.03334   Median :0.04003  
    ##  Mean   :  6108.8           Mean   :0.17797   Mean   :0.13932   Mean   :0.23204  
    ##  3rd Qu.:  5200.0           3rd Qu.:0.23223   3rd Qu.:0.14384   3rd Qu.:0.38593  
    ##  Max.   :351400.0           Max.   :0.92699   Max.   :0.91998   Max.   :0.92000  
    ##      LDA_03            LDA_04        global_rate_positive_words
    ##  Min.   :0.01818   Min.   :0.01819   Min.   :0.00000           
    ##  1st Qu.:0.02547   1st Qu.:0.02857   1st Qu.:0.02740           
    ##  Median :0.04000   Median :0.04001   Median :0.03805           
    ##  Mean   :0.22978   Mean   :0.22089   Mean   :0.03870           
    ##  3rd Qu.:0.39191   3rd Qu.:0.36374   3rd Qu.:0.04957           
    ##  Max.   :0.92554   Max.   :0.92000   Max.   :0.13308           
    ##  global_rate_negative_words avg_positive_polarity avg_negative_polarity
    ##  Min.   :0.00000            Min.   :0.0000        Min.   :-1.0000      
    ##  1st Qu.:0.01010            1st Qu.:0.3068        1st Qu.:-0.3298      
    ##  Median :0.01562            Median :0.3583        Median :-0.2587      
    ##  Mean   :0.01715            Mean   :0.3526        Mean   :-0.2622      
    ##  3rd Qu.:0.02227            3rd Qu.:0.4111        3rd Qu.:-0.1889      
    ##  Max.   :0.13693            Max.   :1.0000        Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity     shares      
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :    22  
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:   976  
    ##  Median :0.1000     Median : 0.00000         Median :  1500  
    ##  Mean   :0.2783     Mean   : 0.06632         Mean   :  3341  
    ##  3rd Qu.:0.5000     3rd Qu.: 0.13636         3rd Qu.:  2800  
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :233400

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

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

*group 2*

``` r
subnews2 <- select(sum_news, 7:12, shares)
corr2 <- cor(subnews2)
corrplot(corr2)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

*group3*

``` r
subnews3 <- select(sum_news, 13:18, shares)
corr3 <- cor(subnews3)
corrplot(corr3)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

*group4*

``` r
subnews4 <- select(sum_news, -c(1:18))
corr4 <- cor(subnews4)
corrplot(corr4)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

*group2*

``` r
ggpairs(subnews2)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

*group3*

``` r
ggpairs(subnews3)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

*group4*

``` r
ggpairs(subnews4)
```

![](fridayAnalysis_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

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
    ## 3990 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 3989, 3989, 3989, 3989, 3989, 3989, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          RMSE      Rsquared      MAE     
    ##   0.01017652  8881.089  0.0011162345  3132.524
    ##   0.01894077  8969.812  0.0018288610  3547.423
    ##   0.02587500  8584.615  0.0005415354  3099.468
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.025875.

**The final chosen model with best tuning parameters**

``` r
regTree$bestTune %>% knitr::kable()
```

|   |       cp |
| - | -------: |
| 3 | 0.025875 |

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
    ## 3990 samples
    ##   30 predictor
    ## 
    ## Pre-processing: centered (30), scaled (30) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 3592, 3590, 3591, 3591, 3591, 3593, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   50      8177.422  0.02071771  2994.546
    ##   1                  100      8183.743  0.02555727  3008.301
    ##   1                  150      8191.879  0.02726473  3009.772
    ##   2                   50      8181.769  0.02878285  2996.172
    ##   2                  100      8206.841  0.03191764  3007.730
    ##   2                  150      8248.156  0.03492136  3032.071
    ##   3                   50      8174.255  0.03883366  3006.696
    ##   3                  100      8214.708  0.04069436  3043.529
    ##   3                  150      8252.472  0.04062304  3066.590
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning
    ##  parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth =
    ##  3, shrinkage = 0.1 and n.minobsinnode = 10.

**The final chosen model with best tuning parameters**

``` r
boostTree$bestTune %>% knitr::kable()
```

|   | n.trees | interaction.depth | shrinkage | n.minobsinnode |
| - | ------: | ----------------: | --------: | -------------: |
| 7 |      50 |                 3 |       0.1 |             10 |

### Test and Compare

``` r
tree_pred <- postResample(predict(regTree, newdata = newsTest), newsTest$shares)
boost_pred <- postResample(predict(boostTree, newdata = newsTest), newsTest$shares)
round(rbind(tree_pred, boost_pred), 4)
```

    ##                RMSE Rsquared      MAE
    ## tree_pred  7162.154       NA 2920.348
    ## boost_pred 7205.558    0.018 2880.542

Smaller root mean square error (RMSE) and smaller mean absolute error
(MAE) shows that we have a better fit of data and more accurate
predictions in test dataset.

Choose the better model with relatively smaller root mean square error
(RMSE), smaller mean absolute error (MAE) and bigger R squared.
