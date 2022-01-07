import matplotlib.pyplot as plt
from argparse import ArgumentParser
from DataModels import FederatedData, UnifiedData
import os


def plot_together(data_dict : dict, title : str, dir : str) :
    for label, data in data_dict.items() :
        plt.plot(data, label=label)
    plt.title(title)
    plt.legend()
    if not os.path.exists(dir+"\\") :
        os.mkdir(dir+"\\")
    plt.savefig(dir+"\\"+title+".png")
    plt.clf()


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
    # Source    : ROUND 1   + Centralised
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[0],
        "Client 2" : f2.losses[0],
        "Client 3" : f3.losses[0],
        "Centralised " : u.val_losses,
    }, "Val Loss - Round 1", args.savedir)

    # Plot 2
    # ===============================================
    # Source    : ROUND 2   + Centralised
    # Data      : loss      + loss
    plot_together({
        "Client 1" : f1.losses[1],
        "Client 2" : f2.losses[1],
        "Client 3" : f3.losses[1],
        "Centralised " : u.val_losses,
    }, "Val Loss - Round 2", args.savedir)

    # Plot 3
    # ===============================================
    # Source    : ROUND 3   + Centralised
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[2],
        "Client 2" : f2.losses[2],
        "Client 3" : f3.losses[2],
        "Centralised " : u.val_losses,
    }, "Val Loss - Round 3", args.savedir)

    # Plot 4
    # ===============================================
    # Source    : ROUND 1   + Centralised
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[0],
        "Client 2" : f2.losses[0],
        "Client 3" : f3.losses[0],
        "Centralised " : u.losses,
    }, "Loss - Round 1", args.savedir)

    # Plot 5
    # ===============================================
    # Source    : ROUND 2   + Centralised
    # Data      : loss      + loss
    plot_together({
        "Client 1" : f1.losses[1],
        "Client 2" : f2.losses[1],
        "Client 3" : f3.losses[1],
        "Centralised " : u.losses,
    }, "Loss - Round 2", args.savedir)

    # Plot 6
    # ===============================================
    # Source    : ROUND 3   + Centralised
    # Data      : loss      + val_loss
    plot_together({
        "Client 1" : f1.losses[2],
        "Client 2" : f2.losses[2],
        "Client 3" : f3.losses[2],
        "Centralised " : u.losses,
    }, "Loss - Round 3", args.savedir)

    # Plot 7
    # ===============================================
    # Source    : ROUND 1   + Centralised
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[0],
        "Client 2" : f2.accuracies[0],
        "Client 3" : f3.accuracies[0],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Round 1", args.savedir)

    # Plot 8
    # ===============================================
    # Source    : ROUND 2   + Centralised
    # Data      : acc       + acc
    plot_together({
        "Client 1" : f1.accuracies[1],
        "Client 2" : f2.accuracies[1],
        "Client 3" : f3.accuracies[1],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Round 2", args.savedir)

    # Plot 9
    # ===============================================
    # Source    : ROUND 3   + Centralised
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[2],
        "Client 2" : f2.accuracies[2],
        "Client 3" : f3.accuracies[2],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Round 3", args.savedir)

    # Plot 10
    # ===============================================
    # Source    : ROUND 1   + Centralised
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[0],
        "Client 2" : f2.accuracies[0],
        "Client 3" : f3.accuracies[0],
        "Centralised " : u.accuracies,
    }, "Acc - Round 1", args.savedir)

    # Plot 11
    # ===============================================
    # Source    : ROUND 2   + Centralised
    # Data      : acc       + acc
    plot_together({
        "Client 1" : f1.accuracies[1],
        "Client 2" : f2.accuracies[1],
        "Client 3" : f3.accuracies[1],
        "Centralised " : u.accuracies,
    }, "Acc - Round 2", args.savedir)

    # Plot 12
    # ===============================================
    # Source    : ROUND 3   + Centralised
    # Data      : acc       + val_acc
    plot_together({
        "Client 1" : f1.accuracies[2],
        "Client 2" : f2.accuracies[2],
        "Client 3" : f3.accuracies[2],
        "Centralised " : u.accuracies,
    }, "Acc - Round 3", args.savedir)
    
    # Plot 13
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Centralised
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f1.losses[0],
        "Round 2" : f1.losses[1],
        "Round 3" : f1.losses[2],
        "Centralised " : u.val_losses,
    }, "Val Loss - Client 1 vs Centralised", args.savedir)

    # Plot 14
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Centralised
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f2.losses[0],
        "Round 2" : f2.losses[1],
        "Round 3" : f2.losses[2],
        "Centralised " : u.val_losses,
    }, "Val Loss - Client 2 vs Centralised", args.savedir)

    # Plot 15
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Centralised
    # Data      : loss                  + val_loss
    plot_together({
        "Round 1" : f3.losses[0],
        "Round 2" : f3.losses[1],
        "Round 3" : f3.losses[2],
        "Centralised " : u.val_losses,
    }, "Val Loss - Client 3 vs Centralised", args.savedir)

    # Plot 16
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Centralised
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f1.losses[0],
        "Round 2" : f1.losses[1],
        "Round 3" : f1.losses[2],
        "Centralised " : u.losses,
    }, "Loss - Client 1 vs Centralised", args.savedir)

    # Plot 17
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Centralised
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f2.losses[0],
        "Round 2" : f2.losses[1],
        "Round 3" : f2.losses[2],
        "Centralised " : u.losses,
    }, "Loss - Client 2 vs Centralised", args.savedir)

    # Plot 18
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Centralised
    # Data      : loss                  + loss
    plot_together({
        "Round 1" : f3.losses[0],
        "Round 2" : f3.losses[1],
        "Round 3" : f3.losses[2],
        "Centralised " : u.losses,
    }, "Loss - Client 3 vs Centralised", args.savedir)
    
    # Plot 19
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Centralised
    # Data      : acc                   + val_acc
    plot_together({
        "Round 1" : f1.accuracies[0],
        "Round 2" : f1.accuracies[1],
        "Round 3" : f1.accuracies[2],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Client 1 vs Centralised", args.savedir)

    # Plot 20
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Centralised
    # Data      : acc                   + val_acc
    plot_together({
        "Round 1" : f2.accuracies[0],
        "Round 2" : f2.accuracies[1],
        "Round 3" : f2.accuracies[2],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Client 2 vs Centralised", args.savedir)

    # Plot 21
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Centralised
    # Data      : acc                   + val_acc 
    plot_together({
        "Round 1" : f3.accuracies[0],
        "Round 2" : f3.accuracies[1],
        "Round 3" : f3.accuracies[2],
        "Centralised " : u.val_accuracies,
    }, "Val Acc - Client 3 vs Centralised", args.savedir)

    # Plot 22
    # ===============================================
    # Source    : Clinet 1 ROUND 1-3    + Centralised
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f1.accuracies[0],
        "Round 2" : f1.accuracies[1],
        "Round 3" : f1.accuracies[2],
        "Centralised " : u.accuracies,
    }, "Acc - Client 1 vs Centralised", args.savedir)

    # Plot 23
    # ===============================================
    # Source    : Clinet 2 ROUND 1-3    + Centralised
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f2.accuracies[0],
        "Round 2" : f2.accuracies[1],
        "Round 3" : f2.accuracies[2],
        "Centralised " : u.accuracies,
    }, "Acc - Client 2 vs Centralised", args.savedir)

    # Plot 24
    # ===============================================
    # Source    : Clinet 3 ROUND 1-3    + Centralised
    # Data      : acc                   + acc
    plot_together({
        "Round 1" : f3.accuracies[0],
        "Round 2" : f3.accuracies[1],
        "Round 3" : f3.accuracies[2],
        "Centralised " : u.accuracies,
    }, "Acc - Client 3 vs Centralised", args.savedir)


if __name__=="__main__":
    parser = ArgumentParser(description="Convert pkl format to plots")
    parser.add_argument("--name1", type=str, required=True, help="Name of federated file 1")
    parser.add_argument("--name2", type=str, required=True, help="Name of federated file 2")
    parser.add_argument("--name3", type=str, required=True, help="Name of federated file 3")
    parser.add_argument("--nameU", type=str, required=True, help="Name of unified file")
    parser.add_argument("--savedir", type=str, required=True, help="Name of save directory plots")
    args = parser.parse_args()
    plot(args)