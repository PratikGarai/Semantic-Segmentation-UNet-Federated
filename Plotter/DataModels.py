import pickle
import re
import matplotlib.pyplot as plt

REG = "epoch [0-9]+ - loss \: ([0-9]+\.[0-9]+) \- acc \: ([0-9]+\.[0-9]+)"
REG2 = "epoch [0-9]+ - loss \: ([0-9]+\.[0-9]+) \- acc \: ([0-9]+\.[0-9]+) \- val loss \: ([0-9]+\.[0-9]+) \- val acc \: ([0-9]+\.[0-9]+)"

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
        ob : FederatedData = pickle.load(fl)
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
                l.append(float(loss))
                a.append(float(acc))
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


    def plot_all_round_loss_in_one_graph(self) :
        for i in range(self.n_rounds) :
            plt.plot(self.losses[i], label=f"Round : {i+1}")
        plt.legend()
        plt.show()
    

    def plot_one_round_loss_in_one_graph(self, n : int) :
        plt.plot(self.losses[n], label=f"Round : {n}")
        plt.legend()
        plt.show()



class UnifiedData : 

    def __init__(self) :
        self.losses : list = []
        self.accuracies : list = []
        self.val_losses : list = []
        self.val_accuracies : list = []
        self.epochs : int = 0
    

    def save(self, filename : str) :
        fl = open(filename+".pkl", "wb")
        pickle.dump(self, fl, pickle.HIGHEST_PROTOCOL)
        fl.close()
    

    def load(self, filename : str) :
        fl = open(filename+".pkl", "rb")
        ob : UnifiedData = pickle.load(fl)
        self.losses = ob.losses
        self.accuracies = ob.accuracies 
        self.val_losses = ob.val_losses
        self.val_accuracies = ob.val_accuracies 
        fl.close()
    

    def read_file(self, filename : str) :
        fl = open(filename, "r")
        l = []
        a = []
        vl = []
        va = []
        for line in fl :
            results = re.search(REG2, line)
            if results : 
                loss, acc, vloss, vacc = results.groups()
                l.append(float(loss))
                a.append(float(acc))
                vl.append(float(vloss))
                va.append(float(vacc))
                self.epochs += 1
        self.losses = l
        self.accuracies = a
        self.val_losses = vl
        self.val_accuracies = va
        fl.close()
    

    def stringify(self) :
        s = ""
        s += f"Epochs         : {self.epochs}\n"
        s += f"Acc  : , {str(self.accuracies)}\n"
        s += f"Loss : , {str(self.losses)}\n"
        return s


    def plot_losses(self) :
        plt.plot(self.losses, label="Loss")
        plt.legend()
        plt.show()
    

    def plot_accuracy(self) :
        plt.plot(self.accuracies, label=f"Accuracy")
        plt.legend()
        plt.show()

    def plot_val_losses(self) :
        plt.plot(self.val_losses, label="Val. Loss")
        plt.legend()
        plt.show()
    

    def plot_val_accuracy(self) :
        plt.plot(self.val_accuracies, label=f"Val. Accuracy")
        plt.legend()
        plt.show()