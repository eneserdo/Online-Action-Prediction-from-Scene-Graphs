import argparse

def arg_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("-mr", "--mask_ready", action="store_true", help="to load already existing mask")
    ap.add_argument("-lo", "--edge_linear_out", type=int, default=32, help="shift in x axis to align depth with rgb")
    ap.add_argument("-ci", "--conv_in", type=int, default=32, help="shift in x axis to align depth with rgb")
    ap.add_argument("-co", "--conv_out", type=int, default=32, help="shift in x axis to align depth with rgb")
    ap.add_argument("-hs", "--hidden_size", type=int, default=128, help="shift in x axis to align depth with rgb")
    ap.add_argument("-bs", "--batch_size", type=int, default=16, help="shift in x axis to align depth with rgb")

    return ap.parse_args()