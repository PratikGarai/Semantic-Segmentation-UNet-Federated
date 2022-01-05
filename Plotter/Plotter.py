import matplotlib.pyplot as plt
from argparse import ArgumentParser
from DataModels import FederatedData

def parse(args : ArgumentParser) :
    f1 = FederatedData()
    f1.read(args.name1)
    f2 = FederatedData()
    f2.read(args.name2)
    f3 = FederatedData()
    f3.read(args.name3)
    

if __name__=="__main__":
    parser = ArgumentParser(description="Convert pkl format to plots")
    parser.add_argument("--name1", type=str, required=True, help="Name of federated file 1")
    parser.add_argument("--name2", type=str, required=True, help="Name of federated file 2")
    parser.add_argument("--name3", type=str, required=True, help="Name of federated file 3")
    args = parser.parse_args()
    parse(args)