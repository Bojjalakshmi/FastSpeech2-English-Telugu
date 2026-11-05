# Entry point for Telugu Preprocessing
import sys
import os
import argparse
sys.path.append('/content/project')
from fastspeech2.prepare_features import main

if __name__ == '__main__':
    # Support passing optional csv through argparse matching your previous cell demands
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=False)
    args, unknown = parser.parse_known_args()
    main(config_path='/content/project/Telugu/config.yaml')
