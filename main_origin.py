import os
import sys
import math
import timeit

import training
import nlp
import classifier
import feature_selection

def compare(file1_name, file2_name):
    agree, disagree = 0, 0

    f1 =  open(file1_path)     
    f2 =  open(file2_path)

    next(f1)
    next(f2)

    for line1, line2 in zip(f1, f2):
        id1_, _, label1 = training.split_doc(line1.strip())
        id2_, _, label2 = training.split_doc(line2.strip())

        assert id1_ == id2_

        if label1 == label2:
            agree += 1
        else:
            disagree += 1

    result = agree*100/(agree+disagree)

    print('-'*60)
    print("Comparison of '{} and '{}'".format(file1_name, file2_name))
    print("Total:", agree+disagree)
    print("Accuracy: {}%".format(result))

    return result

def validate_argv():
    start = 0
    training_flag = True
    predic_fun, feature_fun = None, None
    feature_num = 0

    if len(sys.argv) == 1:
        training_flag = True

    elif len(sys.argv) == 2:
        if sys.argv[1] == "True":
            training_flag = True
        elif sys.argv[1] == "False":
            training_flag = False

    elif len(sys.argv) == 4:
        start = 1
        training_flag = True


    elif len(sys.argv) == 5:
        start = 2
        if sys.argv[1] == "True":
            training_flag = True
        elif sys.argv[1] == "False":
            training_flag = False

    if len(sys.argv) == 1 or len(sys.argv) == 2:
        predict_fun = classifier.multinomial
        feature_fun = feature_selection.kai_square
        feature_num = 40000
    else:
        if sys.argv[start] == "multi":
            predict_fun = classifier.multinomial
        elif sys.argv[start] == "bern":
            predict_fun = classifier.bernoulli
        else:
            print("[ERRPR] Wrong model in argv[1]")
            sys.exit(-1)

        if sys.argv[start+1] == "None":
            feature_fun = None
        elif sys.argv[start+1] == "doc":
            feature_fun = feature_selection.doc_frequency
        elif sys.argv[start+1] == "term":
            feature_fun = feature_selection.term_frequency
        elif sys.argv[start+1] == "mutual":
            feature_fun = feature_selection.mutual_information
        elif sys.argv[start+1] == "kai":
            feature_fun = feature_selection.kai_square
        else:
            print("[ERROR] Wrong feature function in argv[2]")
            
        if sys.argv[start+1] != "None":
            feature_num = int(sys.argv[start+2])
        else:
            feature_num = None
    
    return training_flag, predict_fun, feature_fun, feature_num

def main():
    training_file = "ratings_train.txt"
    query_file = "ratings_valid.txt"
    result_file = "ratings_valid_result.txt"

    start_time = timeit.default_timer()

    training_flag, predict_fun, feature_fun, feature_num  = validate_argv()
    
    dic = None
    if training_flag:
        dic = training.train(training_file)
    else:
        import pickle
        with open("data_training.dat", "rb") as f:
            dic = pickle.load(f)

    if feature_fun != None:
        feature_selection.select_feature(dic, feature_num, feature_fun)
    else:
        print("-"*60)
        print("No feature selection!")
  
    classifier.nbc(query_file, result_file, dic, predict_fun)

    end_time = timeit.default_timer()
    print("Total time: {:.3} sec".format(end_time - start_time))

    result = compare("ratings_valid.txt", result_file)

if __name__ == "__main__":
    main()

