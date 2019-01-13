import csv
import os
import sys
import copy
import ntpath
import math
import timeit
import argparse
import pickle

import training
import nlp
import classifier
import feature_selection

import matplotlib.pyplot as plt
import matplotlib.pylab as plb

training_file = "ratings_train.txt"
query_file = "ratings_valid.txt"
result_file = "ratings_valid_result.txt"
training_data_file = "data_training.dat"
csv_file = "test_result.csv"

training_path = None
query_path = None
result_path = None
training_data_path = None
csv_path = None

models = [classifier.multinomial, classifier.bernoulli]
features = [None, feature_selection.doc_frequency, feature_selection.term_frequency,
            feature_selection.mutual_information, feature_selection.kai_square]
counts = [1000, 5000, 10000, 20000,
          30000, 40000, 50000]

def compare():
    if os.path.exists(training_data_path):
        with open(training_data_path, "rb") as f:
            dic = pickle.load(f)
    else:
        dic = training.train(training_path)
        with open(training_data_path, "wb") as f:
            pickle.dump(dic, f)

    run(dic)
    multi, bern = csv_reader()
    plotter(multi, bern)

def run(dic_origin):
    csv_f = open(csv_path, "w")
    writer = csv.writer(csv_f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_ALL)

    for predict_fun in models:
        for feature_fun in features:
                if feature_fun == None:
                    dic = copy.deepcopy(dic_origin)

                    classifier.nbc(query_path, result_path, dic, predict_fun)
                    result = accuracy(query_path, result_path)
                    writer.writerow([predict_fun.__name__, "None",
                                     "None", result])
                else:
                    for feature_num in counts:
                        dic = copy.deepcopy(dic_origin)

                        feature_selection.select_feature(dic, feature_num, feature_fun)
          
                        classifier.nbc(query_path, result_path, dic, predict_fun)
                        result = accuracy(query_path, result_path)
                        writer.writerow([predict_fun.__name__, feature_fun.__name__,
                                         feature_num, result])
    csv_f.close()

def csv_reader():
    multi = {"None":[],"mutual_information":[],
             "term_frequency":[], "doc_frequency":[],
             "kai_square":[]}
    bern = {"None":[],"mutual_information":[],
            "term_frequency":[], "doc_frequency":[],
            "kai_square":[]}

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if row[0] == "multinomial":
                multi[row[1]].append(float(row[3]))
            elif row[0] == "bernoulli":
                bern[row[1]].append(float(row[3]))

    return multi, bern

def plotter(multi, bern):
    name = "multinomial model"
    model = multi
    plt.subplot(1, 2, 1)
    plt.plot(counts, model["None"]*7,
             counts, model["doc_frequency"],
             counts, model["term_frequency"],
             counts, model["mutual_information"],
             counts, model["kai_square"])
    plt.title(name)
    plt.xlabel("number of feature")
    plt.ylabel("accuracy")
    plb.legend(["no select", "doc", "term", "mutual", "kai"], loc='best')

    name = "bernoulli model"
    model = bern
    plt.subplot(1, 2, 2)
    plt.plot(counts, model["None"]*7,
             counts, model["doc_frequency"],
             counts, model["term_frequency"],
             counts, model["mutual_information"],
             counts, model["kai_square"])
    plt.title(name)
    plt.xlabel("number of feature")
    plt.ylabel("accuracy")
    plb.legend(["no select", "doc", "term", "mutual", "kai"], loc='best')

    plt.show()

def accuracy(file1_path, file2_path):
    agree, disagree = 0, 0

    file1_name = ntpath.basename(file1_path)
    file2_name = ntpath.basename(file2_path)

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

def handle_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model' ,metavar='model',type=str,
        help="choose classifying moedel between 'multi'(multinomial) and 'bern'(bernoulli)")
    group.add_argument('-c', '--compare', action='store_true', help='compare several model, feature selection function and number of features')

    parser.add_argument('-f', '--feature', metavar='feature', type=str,
        help="choose feature selection alorithm between 'df'(document frequency), 'tf'(term frequency), 'mi'(mutual information) and 'ks'(kai square).")
    parser.add_argument('-n', '--num_feature', type=int, help="choose number of features")
    parser.add_argument('-l', '--load', action='store_true', help="load pre-trained file")


    args = parser.parse_args()

    if args.compare:
        predict_fun = None
    elif args.model == "multi":
        predict_fun = classifier.multinomial
    elif args.model == "bern":
        predict_fun = classifier.bernoulli
    else:
        parser.error("choose model between 'multi' and 'bern'")

    if args.compare:
        feature_fun = None
    elif args.feature == "df":
        feature_fun = feature_selection.doc_frequency
    elif args.feature == "tf":
        feature_fun = feature_selection.term_frequency
    elif args.feature == "mi":
        feature_fun = feature_selection.mutual_information
    elif args.feature == "ks":
        feature_fun = feature_selection.kai_square
    elif args.feature == None:
        feature_fun = None
    else:
        parser.error("choose model between 'df, 'tf', 'mi', and 'ks'")

    if not args.feature and args.num_feature:
        parser.error("To use -n option, select -f option first")

    if args.feature and not args.num_feature:
        parser.error("To use -f option, select -n option")
    if args.compare and (args.feature or args.num_feature or args.load):
        parser.error("-c option must be used alone")
    
    return args, predict_fun, feature_fun

def main():
    global training_path, query_path, result_path, training_data_path, csv_path
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, "data")
    
    training_path = os.path.join(data_path, training_file)
    query_path = os.path.join(data_path, query_file)
    result_path = os.path.join(data_path, result_file)
    training_data_path = os.path.join(data_path, training_data_file)
    csv_path = os.path.join(data_path, csv_file)

    start_time = timeit.default_timer()

    args, predict_fun, feature_fun = handle_args()
    load_flag = args.load
    feature_num = args.num_feature
    compare_flag = args.compare

    if compare_flag:
        compare()
    else:
        dic = None
        if load_flag:
            with open(training_data_path, "rb") as f:
                dic = pickle.load(f)
        else:
            dic = training.train(training_path)
            with open(training_data_path, "wb") as f:
                pickle.dump(dic, f)

        if feature_fun != None:
            feature_selection.select_feature(dic, feature_num, feature_fun)
        else:
            print("-"*60)
            print("No feature selection!")
      
        classifier.nbc(query_path, result_path, dic, predict_fun)

        end_time = timeit.default_timer()
        print("Total time: {:.3} sec".format(end_time - start_time))

        result = accuracy(query_path, result_path)

if __name__ == "__main__":
    main()

