import argparse


def arg_parser():

    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-p", "--process_it", action="store_true", help="...")
    ap.add_argument("-bs", "--batch_size", type=int, default=64, help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")

    # Model
    # ap.add_argument("-nef", "--num_edge_feat", type=int, default=32, help="...")
    # ap.add_argument("-ci", "--conv_in", type=int, default=32, help="...")
    ap.add_argument("-ch", "--conv_hidden", type=int, default=64, help="...")
    ap.add_argument("-co", "--conv_out", type=int, default=64, help="...")
    ap.add_argument("-hs", "--hidden_size", type=int, default=128, help="...")
    ap.add_argument("-r", "--restore", type=int, help="...")

    # Logging
    ap.add_argument("-v", "--use_visdom", action="store_true", help="...")

    return ap.parse_args()



def arg_parser_inference():

    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--folder_path", required=True, type=str, help="...")    
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")

    ap.add_argument("-gt", "--ground_truth_path", help="...")

    # Model
    ap.add_argument("-ch", "--conv_hidden", type=int, default=64, help="...")
    ap.add_argument("-co", "--conv_out", type=int, default=64, help="...")
    ap.add_argument("-hs", "--hidden_size", type=int, default=128, help="...")
    ap.add_argument("-r", "--restore", type=int, required=True, help="...")

    return ap.parse_args()