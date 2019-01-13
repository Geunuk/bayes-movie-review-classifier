import os
import copy
import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

from main import accuracy
from classifier import nbc, multinomial, bernoulli
from feature_selection import select_feature, doc_frequency
from feature_selection import term_frequency, mutual_information, kai_square

train_file = "ratings_train.txt"
query_file = "ratings_valid.txt"
result_file = "ratings_valid_result.txt"



training_file = "ratings_train.txt"
query_file = "ratings_valid.txt"
result_file = "ratings_valid_result.txt"
csv_file = "test_result.csv"

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, "data")
training_path = os.path.join(data_path, training_file)
query_path = os.path.join(data_path, query_file)
result_path = os.path.join(data_path, result_file)
csv_path = os.path.join(data_path, csv_file)

models = [multinomial, bernoulli]
features = [None, doc_frequency, term_frequency,
            mutual_information, kai_square]
counts = [1000, 5000, 10000, 20000,
          30000, 40000, 50000]

def compare(training_path, query_path, result_path, csv_path):
    run(training_path, query_path, result_path, csv_path)
    multi, bern = csv_reader(csv_path)
    subplotter(multi, bern)

def run():
    csv_f = open(csv_path, "w")

    writer = csv.writer(csv_f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_ALL)
    training_file = open("data/data_training.dat", "rb")
    dic_origin = pickle.load(training_file)

    for predict_fun in models:
        for feature_fun in features:
                if feature_fun == None:
                    dic = copy.deepcopy(dic_origin)

                    nbc(query_path, result_path, dic, predict_fun)
                    result = accuracy(query_path, result_path)
                    writer.writerow([predict_fun.__name__, "None",
                                     "None", result])
                else:
                    for feature_num in counts:
                        dic = copy.deepcopy(dic_origin)

                        select_feature(dic, feature_num, feature_fun)
          
                        nbc(query_path, result_path, dic, predict_fun)
                        result = accuracy(query_path, result_path)
                        writer.writerow([predict_fun.__name__, feature_fun.__name__,
                                         feature_num, result])
    training_file.close()
    csv_file.close()

def csv_reader(csv_path):
    multi = {"None":[],"mutual_information":[],
             "term_frequency":[], "doc_frequency":[],
             "kai_square":[]}
    bern = {"None":[],"mutual_information":[],
            "term_frequency":[], "doc_frequency":[],
            "kai_square":[]}

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
       
        for row in reader:
            print(row)
            if row[0] == "multinomial":
                multi[row[1]].append(float(row[3]))
            elif row[0] == "bernoulli":
                bern[row[1]].append(float(row[3]))
    return multi, bern

def plotter(model, name):
    plt.plot(counts, model["None"]*7,
             counts, model["doc_frequency"],
             counts, model["term_frequency"],
             counts, model["mutual_information"],
             counts, model["kai_square"])
    plt.title(name)
    plt.xlabel("number of feature")
    plt.ylabel("accuracy")
    plb.legend(["no select", "doc", "term", "mutual", "kai"], loc='best')

    plt.savefig(name + ".jpg")
    plt.show()

def plotter(mutli, bern):
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

if __name__ == "__main__":

    #test(file_name)

    multi, bern = csv_reader(file_name)
    subplotter(multi, bern)
    #plotter(multi, "multinomial model")
    #plotter(bern, "bernoulli model")
