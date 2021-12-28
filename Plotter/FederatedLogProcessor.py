from argparse import ArgumentParser
from DataModels import FederatedData

def parse(args : ArgumentParser) :
    f = FederatedData()
    f.read_file(args.file, args.sep)
    return

if __name__=="__main__":
    parser = ArgumentParser(description="Convert a federated training log to pkl format")
    parser.add_argument("--file", type=str, required=True, help="Name of the txt file")
    parser.add_argument("--mode", type=str, required=True, choices=["F", "U"], help="Mode of file, U for unified, F for federated")
    parser.add_argument("--sep", type=str, help="Separator of federated rounds in \"quotes\". Required in mode F")
    args = parser.parse_args()
    parse(args)