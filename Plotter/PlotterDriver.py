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
    # Data      : loss      + loss
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

    # Plot 4
    # ===============================================
    # Source    : ROUND 1   + Unified
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[0],
        "Client 2" : f2.losses[0],
        "Client 3" : f3.losses[0],
        "Unified " : u.losses,
    }, "Loss - Round 1")

    # Plot 5
    # ===============================================
    # Source    : ROUND 2   + Unified
    # Data      : loss      + loss
    plot_together({
        "Client 1" : f1.losses[1],
        "Client 2" : f2.losses[1],
        "Client 3" : f3.losses[1],
        "Unified " : u.losses,
    }, "Loss - Round 2")

    # Plot 6
    # ===============================================
    # Source    : ROUND 3   + Unified
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[2],
        "Client 2" : f2.losses[2],
        "Client 3" : f3.losses[2],
        "Unified " : u.losses,
    }, "Loss - Round 3")

    # Plot 7
    # ===============================================
    # Source    : ROUND 1   + Unified
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[0],
        "Client 2" : f2.accuracies[0],
        "Client 3" : f3.accuracies[0],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Round 1")

    # Plot 8
    # ===============================================
    # Source    : ROUND 2   + Unified
    # Data      : acc       + acc
    plot_together({
        "Client 1" : f1.accuracies[1],
        "Client 2" : f2.accuracies[1],
        "Client 3" : f3.accuracies[1],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Round 2")

    # Plot 9
    # ===============================================
    # Source    : ROUND 3   + Unified
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[2],
        "Client 2" : f2.accuracies[2],
        "Client 3" : f3.accuracies[2],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Round 3")

    # Plot 10
    # ===============================================
    # Source    : ROUND 1   + Unified
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[0],
        "Client 2" : f2.accuracies[0],
        "Client 3" : f3.accuracies[0],
        "Unified " : u.accuracies,
    }, "Acc - Round 1")

    # Plot 11
    # ===============================================
    # Source    : ROUND 2   + Unified
    # Data      : acc       + acc
    plot_together({
        "Client 1" : f1.accuracies[1],
        "Client 2" : f2.accuracies[1],
        "Client 3" : f3.accuracies[1],
        "Unified " : u.accuracies,
    }, "Acc - Round 2")

    # Plot 12
    # ===============================================
    # Source    : ROUND 3   + Unified
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[2],
        "Client 2" : f2.accuracies[2],
        "Client 3" : f3.accuracies[2],
        "Unified " : u.accuracies,
    }, "Acc - Round 3")
    
    # Plot 13
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Unified
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f1.losses[0],
        "Round 2" : f1.losses[1],
        "Round 3" : f1.losses[2],
        "Unified " : u.val_losses,
    }, "Val Loss - Client 1 vs Unified")

    # Plot 14
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Unified
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f2.losses[0],
        "Round 2" : f2.losses[1],
        "Round 3" : f2.losses[2],
        "Unified " : u.val_losses,
    }, "Val Loss - Client 2 vs Unified")

    # Plot 15
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Unified
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f3.losses[0],
        "Round 2" : f3.losses[1],
        "Round 3" : f3.losses[2],
        "Unified " : u.val_losses,
    }, "Val Loss - Client 3 vs Unified")

    # Plot 16
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Unified
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f1.losses[0],
        "Round 2" : f1.losses[1],
        "Round 3" : f1.losses[2],
        "Unified " : u.losses,
    }, "Loss - Client 1 vs Unified")

    # Plot 17
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Unified
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f2.losses[0],
        "Round 2" : f2.losses[1],
        "Round 3" : f2.losses[2],
        "Unified " : u.losses,
    }, "Loss - Client 2 vs Unified")

    # Plot 18
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Unified
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f3.losses[0],
        "Round 2" : f3.losses[1],
        "Round 3" : f3.losses[2],
        "Unified " : u.losses,
    }, "Loss - Client 3 vs Unified")
    
    # Plot 19
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Unified
    # Data      : acc                   + val_acc
    plot_together({
        "Round 1" : f1.accuracies[0],
        "Round 2" : f1.accuracies[1],
        "Round 3" : f1.accuracies[2],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Client 1 vs Unified")

    # Plot 20
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Unified
    # Data      : acc                   + val_acc
    plot_together({
        "Round 1" : f2.accuracies[0],
        "Round 2" : f2.accuracies[1],
        "Round 3" : f2.accuracies[2],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Client 2 vs Unified")

    # Plot 21
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Unified
    # Data      : acc                   + val_acc 
    plot_together({
        "Round 1" : f3.accuracies[0],
        "Round 2" : f3.accuracies[1],
        "Round 3" : f3.accuracies[2],
        "Unified " : u.val_accuracies,
    }, "Val Acc - Client 3 vs Unified")

    # Plot 22
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Unified
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f1.accuracies[0],
        "Round 2" : f1.accuracies[1],
        "Round 3" : f1.accuracies[2],
        "Unified " : u.accuracies,
    }, "Acc - Client 1 vs Unified")

    # Plot 23
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Unified
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f2.accuracies[0],
        "Round 2" : f2.accuracies[1],
        "Round 3" : f2.accuracies[2],
        "Unified " : u.accuracies,
    }, "Acc - Client 2 vs Unified")

    # Plot 24
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Unified
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f3.accuracies[0],
        "Round 2" : f3.accuracies[1],
        "Round 3" : f3.accuracies[2],
        "Unified " : u.accuracies,
    }, "Acc - Client 3 vs Unified")


if __name__=="__main__":
    parser = ArgumentParser(description="Convert pkl format to plots")
    parser.add_argument("--name1", type=str, required=True, help="Name of federated file 1")
    parser.add_argument("--name2", type=str, required=True, help="Name of federated file 2")
    parser.add_argument("--name3", type=str, required=True, help="Name of federated file 3")
    parser.add_argument("--nameU", type=str, required=True, help="Name of unified file")
    args = parser.parse_args()
    plot(args)