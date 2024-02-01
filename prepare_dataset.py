import argparse
import os
from dataset import download_bimanual_dataset, process_bimanual_dataset


def arg_parser_dataset():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default=".", help="Root directory of dataset.")
    ap.add_argument("-d", "--download", action="store_true", help="download dataset")
    ap.add_argument("--fix_fps", action="store_true", help="Fix fps of the videos to 30 fps")
    ap.add_argument("--fix_null", action="store_true", help="Fix null actions in the video of subject_6-task_4-take_1")
    ap.add_argument("-tl", "--temporal_length", type=int, required=True, help="Temporal length of the input")

    return ap.parse_args()


def main():
    args = arg_parser_dataset()

    if args.fix_fps:
        raise NotImplementedError("Fix fps may be broken")

    raw_dir = os.path.join(args.root, "raw")

    if args.download:
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        download_bimanual_dataset(raw_dir, args.fix_null)
        print("Dataset downloaded!")

    process_bimanual_dataset(args.root, args.temporal_length, args.fix_fps)
    print("Dataset processed!")
    print("You can now train your model!")


if __name__ == "__main__":
    main()
