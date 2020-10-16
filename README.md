# Project2

## Required Packages

Please notice that the packages`rmarkdown`, `caret` , `corrplot`, `knitr` and `tidyverse` are required to be installed and loaded before running the code to create the reports. 

## Reports list

The analysis for [Monday is available here](mondayAnalysis.md).

The analysis for [Tuesday is available here](tuesdayAnalysis.md).

The analysis for [Wednesday is available here](wednesdayAnalysis.md).

The analysis for [Thursday is available here](thursdayAnalysis.md).

The analysis for [Friday is available here](fridayAnalysis.md).

The analysis for [Saturday is available here](saturdayAnalysis.md).

The analysis for [Sunday is available here](sundayAnalysis.md).

## The code used to automate the process

```{r, include=TRUE, eval=FALSE, echo=FALSE}
day <- unique(np$weekday)
output_file <- paste0(day, "Analysis.md")

params = lapply(day, FUN = function(x){list(weekday = x)})

reports <- tibble(output_file, params)

library(rmarkdown)

apply(reports, MARGIN = 1, FUN = function(x){

  render(input = "st558Project2.Rmd", output_file = x[[1]], params = x[[2]])
  
  })
  
```
