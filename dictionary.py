import sys

class Dictionary():
    def __init__(self):
        self.cls = {"0":0, "1":0}
        self.repeat = {"doc":{"0":0, "1":0}, "term":{}}
        self.once = {"doc":{"0":0, "1":0}, "term":{}}

    def total_doc(self):
        result = 0
        for c in self.cls:
            result += self.cls[c]
        return result

    def doc_frequency(self, label):
        return self.cls[label]/self.total_doc()
    
    def increment_label(self, label):
        for c in self.cls:
            if label == c:
                self.cls[c] +=1
                break
        else:
            print("[ERROR] Wrong label in increment_label()")
            sys.exit(-1)

    def add_term_repeat(self, term, label):
        if term not in self.repeat["term"]:
            self.repeat["term"][term] = {"0":0, "1":0}

        for c in self.cls:
            if label == c:
                self.repeat["doc"][label] += 1
                self.repeat["term"][term][label] += 1
                break
        else:
            print("[ERROR] wrong label in add_term()")
            sys.exit(-1)

           
    def add_term_once(self, term, label):
        if term not in self.once["term"]:
            self.once["term"][term] = {"0":0, "1":0}

        for c in self.cls:
            if label == c:
                self.once["doc"][label] += 1
                self.once["term"][term][label] += 1
                break
        else:
            print("[ERROR] Wrong label in add_term()")
            sys.exit(-1)
    
if __name__ == '__main__':
    ...
