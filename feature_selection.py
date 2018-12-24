import math
import heapq

def select_feature(dic, cnt, fun):
    print("-"*60)
    print("Start feature selection...")
    print("Feature selection using", fun.__name__)
    print("Select {} terms".format(cnt))

    term_list = []
    remove_list = []
    before_len = len(dic.once["term"])

    for t in dic.once["term"]:
        heapq.heappush(term_list, (fun(dic, t), t))

    for i in range(before_len-cnt):
        remove_list.append(heapq.heappop(term_list)[1])

    for t in remove_list:
        del dic.once["term"][t]
        del dic.repeat["term"][t]

    after_len = len(dic.once["term"])

    print("End feature selection!")
    print("Len(term) : {} -> {}:".format(before_len, after_len))

def doc_frequency(dic, term):
    return (dic.once["term"][term]["0"] + dic.once["term"][term]["1"] )

def term_frequency(dic, term):
    return (dic.repeat["term"][term]["0"] + dic.repeat["term"][term]["1"] )

def mutual_information(dic, term):
    n = dic.cls["0"] + dic.cls["1"] + 4
    n11 = dic.once["term"][term]["0"] + 1
    n10 = dic.once["term"][term]["1"] + 1
    n01 = dic.once["doc"]["0"] - n11 + 1
    n00 = dic.once["doc"]["1"] - n10 + 1

    return (n11*math.log2(n*n11/((n11+n10)*(n11+n01)))
          + n01*math.log2(n*n01/((n00+n01)*(n01+n11)))
          + n10*math.log2(n*n10/((n10+n11)*(n00+n10)))
          + n00*math.log2(n*n00/((n00+n01)*(n00+n10))))/n

def kai_square(dic, term):
    n = dic.cls["0"] + dic.cls["1"]
    n11 = dic.once["term"][term]["0"]
    n10 = dic.once["term"][term]["1"]
    n01 = dic.once["doc"]["0"] - n11
    n00 = dic.once["doc"]["1"] - n10

    return (n*(n11*n00-n10*n01)**2)/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00))
