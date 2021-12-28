import pickle

class FederatedData : 
    def __init__(self) :
        self.n_rounds : int = 0
        self.losses : list = []
        self.accuracies : list = []
        self.round_counts : list = []
    
    def save(self, filename : str) :
        fl = open(filename, "wb")
        pickle.dump(self, fl, pickle.HIGHEST_PROTOCOL)
        fl.close()
    
    def load(self, filename : str) :
        fl = open(filename, "rb")
        ob : FederatedData = pickle.load(fl, pickle.HIGHEST_PROTOCOL)
        self.n_rounds = ob.n_rounds
        self.losses = ob.losses
        self.accuracies = ob.accuracies 
        self.round_counts = ob.round_counts
        fl.close()
    
    def read_file(self, filename : str, sep : str) :
        fl = open(filename, "r")
        for line in fl :
            # print(line)
            pass
        fl.close()
        print(sep)