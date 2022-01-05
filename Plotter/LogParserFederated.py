from argparse import ArgumentParser
from DataModels import FederatedData

def parse(args : ArgumentParser) :
    f = FederatedData()
    f.read_file(args.file, args.sep)
    f.save(args.name)

if __name__=="__main__":
    parser = ArgumentParser(description="Convert a federated training log to pkl format")
    parser.add_argument("--file", type=str, required=True, help="Name of the txt file")
    parser.add_argument("--sep", type=str, help="Separator of federated rounds in \"quotes\". Required in mode F")
    parser.add_argument("--name", type=str, required=True, help="Name with which file is to be saved (exclude extensions)")
    args = parser.parse_args()
    parse(args)