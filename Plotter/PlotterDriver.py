import matplotlib.pyplot as plt
from argparse import ArgumentParser
from DataModels import FederatedData, UnifiedData

def plot_together(data_dict : dict, title : str) :
    for label, data in data_dict.items() :
        plt.plot(data, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot(args : ArgumentParser) :
    f1 = FederatedData()
    f1.load(args.name1)
    f2 = FederatedData()
    f2.load(args.name2)
    f3 = FederatedData()
    f3.load(args.name3)
    u = UnifiedData()
    u.load(args.nameU)

    # Plot 1
    # ===============================================
    # Source    : ROUND 1   + Unified
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[0],
        "Client 2" : f2.losses[0],
        "Client 3" : f3.losses[0],
        "Unified " : u.val_losses,
    }, "Val Loss - Round 1")

    # Plot 2
    # ===============================================
    # Source    : ROUND 2   + Unified
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[1],
        "Client 2" : f2.losses[1],
        "Client 3" : f3.losses[1],
        "Unified " : u.val_losses,
    }, "Val Loss - Round 2")

    # Plot 3
    # ===============================================
    # Source    : ROUND 3   + Unified
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[2],
        "Client 2" : f2.losses[2],
        "Client 3" : f3.losses[2],
        "Unified " : u.val_losses,
    }, "Val Loss - Round 3")


if __name__=="__main__":
    parser = ArgumentParser(description="Convert pkl format to plots")
    parser.add_argument("--name1", type=str, required=True, help="Name of federated file 1")
    parser.add_argument("--name2", type=str, required=True, help="Name of federated file 2")
    parser.add_argument("--name3", type=str, required=True, help="Name of federated file 3")
    parser.add_argument("--nameU", type=str, required=True, help="Name of unified file")
    args = parser.parse_args()
    plot(args)