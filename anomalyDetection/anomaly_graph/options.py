import argparse
parser = argparse.ArgumentParser(description='RTFM')


parser.add_argument('--load-graph', help="Does load a anomaly graphy alread saved as anomaly_graphy.npy in root folder?", default="0")
parser.add_argument('--video-live', help="Test with only 1 file on test set and outputs the detection score for each png", default="-1")