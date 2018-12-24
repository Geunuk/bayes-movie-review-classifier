import math
import sys
import os

import nlp
import training

def nbc(query_file, result_file, dic, predict_fun):
    print("-"*60)
    print("Start Classification...")
    print("Classification using", predict_fun.__name__)
    if predict_fun == bernoulli:
        bernoulli.tmp_result = {"0":0, "1":0}
        for t in dic.once["term"]:
            tmp = bernoulli.tmp_result
            tmp["0"] += math.log(1-smoothing_bernoulli(dic, t, "0"))
            tmp["1"] += math.log(1-smoothing_bernoulli(dic, t, "1"))

    in_f =  open(query_file, "rt")
    out_f = open(result_file, "wt")

    out_f.write(in_f.readline())

    for i, line in enumerate(in_f):
        #if i%1000 == 0:
        #    print(i)

        id_, sentence, label = training.split_doc(line.strip())
        term_list = nlp.parser(sentence)
        new_label = classify(term_list, dic, predict_fun)

        if line.strip()[-1] in ("0", "1"):
            new_line = line.strip()[:-1] + new_label + "\n"
        else:
            new_line = line.strip() + "\t" + new_label + "\n"

        out_f.write(new_line)
    in_f.close()
    out_f.close()
    print("Finish Classification!")

def classify(query_term_list, dic, predict_fun):
    predict0 = predict_fun(query_term_list, dic, "0")
    predict1 = predict_fun(query_term_list, dic, "1")

    if predict0 > predict1:
        return "0"
    else:
        return "1"

def smoothing_multinomial(dic, term, label):
    return (dic.repeat["term"][term][label]+1)/(dic.repeat["doc"][label]+len(dic.repeat["term"]))

def smoothing_bernoulli(dic, term, label):
    return (dic.once["term"][term][label]+1)/(dic.cls[label]+len(dic.cls))

def multinomial(term_list, dic, label):
    result = math.log(dic.doc_frequency(label))

    for t in term_list:
        try:
            result += math.log(smoothing_multinomial(dic, t, label))
        except KeyError:
            continue
      
    return result

def bernoulli(term_list, dic, label):
    result = bernoulli.tmp_result[label]
    result += math.log(dic.doc_frequency(label))

    for t in term_list:
        try:
            result -= math.log(1-smoothing_bernoulli(dic, t, label))
            result += math.log(smoothing_bernoulli(dic, t, label))
        except KeyError:
            continue

    return result
