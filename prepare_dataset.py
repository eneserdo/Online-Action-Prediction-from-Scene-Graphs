import argparse
from dataset import BimanualDataset


def arg_parser_dataset():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default='.', help="Root directory of dataset.")
    ap.add_argument("-contact", "--contact_only", action="store_true", help="...")
    ap.add_argument("--fix_fps", action="store_true", help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=20, help="...")
    ap.add_argument("--pred_length", type=int, default=10, help="...")
    ap.add_argument("--pred_dim", type=int, default=3, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")

    return ap.parse_args()


def main():
    args=arg_parser_dataset()

    # Initiating a single instance will download and process the dataset
    _ = BimanualDataset(root=args.root, 
                        is_processed=False, 
                        split="train", 
                        test_subject=args.test_subject, 
                        validation_take=args.validation_take,
                        temporal_length=args.temporal_length,
                        pred_length=args.pred_length, 
                        contact_only=args.contact_only,
                        fix_fps=args.fix_fps)
    
    print("Dataset preparation is done! You can now train your model")
    

if __name__ == "__main__":
    main()    
