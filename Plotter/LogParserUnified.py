from argparse import ArgumentParser
from DataModels import UnifiedData

def parse(args : ArgumentParser) :
    f = UnifiedData()
    f.read_file(args.file)
    f.save(args.name)

if __name__=="__main__":
    parser = ArgumentParser(description="Convert a unified training log to pkl format")
    parser.add_argument("--file", type=str, required=True, help="Name of the txt file")
    parser.add_argument("--name", type=str, required=True, help="Name with which file is to be saved (exclude extensions)")
    args = parser.parse_args()
    parse(args)