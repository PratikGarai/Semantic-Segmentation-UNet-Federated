import pickle
import re

REG = "epoch [0-9]+ - loss \: ([0-9]+\.[0-9]+) \- acc \: ([0-9]+\.[0-9]+)"

class FederatedData : 

    def __init__(self) :
        self.n_rounds : int = 0
        self.losses : list = []
        self.accuracies : list = []
        self.round_counts : list = []
    

    def save(self, filename : str) :
        fl = open(filename+".pkl", "wb")
        pickle.dump(self, fl, pickle.HIGHEST_PROTOCOL)
        fl.close()
    

    def load(self, filename : str) :
        fl = open(filename+".pkl", "rb")
        ob : FederatedData = pickle.load(fl, pickle.HIGHEST_PROTOCOL)
        self.n_rounds = ob.n_rounds
        self.losses = ob.losses
        self.accuracies = ob.accuracies 
        self.round_counts = ob.round_counts
        fl.close()
    

    def read_file(self, filename : str, sep : str) :
        if not sep : 
            raise Exception("No seperator in Federated mode")
        fl = open(filename, "r")
        l = None
        a = None
        for line in fl :
            if line.strip()==sep :
                if self.n_rounds != 0 :
                    self.losses.append(l)
                    self.accuracies.append(a)
                    self.round_counts.append(len(l))
                self.n_rounds += 1
                l = []
                a = []
            results = re.search(REG, line)
            if results : 
                loss, acc = results.groups()
                l.append(loss)
                a.append(acc)
        self.losses.append(l)
        self.accuracies.append(a)
        self.round_counts.append(len(l))
        fl.close()
    

    def stringify(self) :
        s = ""
        s += f"Rounds          : {self.n_rounds}\n"
        s += f"Round counts    : {str(self.round_counts)}\n"
        for i in range(self.n_rounds) :
            s += f"Acc  : , {str(self.accuracies[i])}\n"
            s += f"Loss : , {str(self.losses[i])}\n"
        
        return s