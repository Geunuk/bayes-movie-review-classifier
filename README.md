# bayes-movie-review-classifier

This program classify movie review into 0 and 1 by using handmade bayes classifier.

## Datasets
Datasets are stored in data directory.
Each dats files are consists of id, review and label.
If review satisfy with movie, label is 1.
If review not satisfy with movie, label is 0.
id, reivew and lable are split by '\t'

* 'ratings_train.txt' - data sets for training
* 'ratings_valid.txt" - data sets for checking validation during development
* 'ratings_valid_result.txt' - file made by classifying review of 'ratings_valid.txt'
* 'ratings_test.txt' -  data sets for test.
But I don't have 'ratings_test_result.txt' file. So use 'ratings_valid.txt' for testing and just ignore this file

## Supported feature selection function
* doc frequency(df)
* term frequency(tf)
* mutual information(mi)
* kasi square(ks)

## Run Algorithm

Select feature function and number of features by -f and -l option
If you once execute the program, trailing result is automatically stored in 'data/data_training.dat' by pickle library
If you want to that file and skip training, use -l option
```
$python3 main.py [-f feature] [-n NUM_FEATURE] [-l]
```

## Compare

Compare several models, feature selection functions, and number of features
```
$python3 main.py -c
```

## Requriements
* <a href="https://matplotlib.org/">matplotlib</a> for plotting result of comparison
* <a href="https://konlpy-ko.readthedocs.io/ko/v0.4.3/">KoNLPy<a/> for parsing Koreans
* <a href="http://konlpy.org/ko/v0.5.1/install/#ubuntu">Mecab<a/> for using when parsing Koreans with KoNLPy

## References
* For feature selection algorithms, look chapter 13 of 'Introduction to Information Retrieval' by Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze.
