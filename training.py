import os
import sys
import timeit

import dictionary
import nlp
import classifier

def split_doc(line):
    id_, _, line = line.partition("\t")
    sentence, _, label = line.partition("\t")
    return id_, sentence, label

def train(file_name):
    print("-"*60)
    print("Start training...")

    start_time = timeit.default_timer()
    dic = dictionary.Dictionary()

    with open(file_name) as f:       
        next(f)
        
        for i, line in enumerate(f):
            #if i%1000 == 0:
            #    print(i)

            id_, sentence, label = split_doc(line.strip())                
            
            dic.increment_label(label)
            term_list = nlp.parser(sentence)
            
            already = set()
            for t in term_list:
                if t not in already:
                    dic.add_term_repeat(t, label)
                    dic.add_term_once(t, label)
                    already.add(t)
                else:
                    dic.add_term_repeat(t, label)
                    
    end_time = timeit.default_timer()
    
    print("End training!")
    print("Training of '{}'".format(file_name))
    print("Training time: {:.3} sec".format(end_time-start_time))
    print("Total {} docs".format(dic.total_doc()))
    print("Good doc:", dic.doc_frequency("1"))
    print("Bad doc:", dic.doc_frequency("0"))
    print("Number of terms:", len(dic.once["term"]))  
    return dic

if __name__ == "__main__":
    file_name = "ratings_train.txt"
    dic = train(file_name)

    import pickle
    with open("data_training.dat", 'wb') as f:
        pickle.dump(dic, f)
