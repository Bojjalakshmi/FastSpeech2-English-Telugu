# Entry point for English Training
import sys
import os
sys.path.append('/content/project')
from fastspeech2.train import main

if __name__ == '__main__':
    main(config_path='/content/project/English/config.yaml')
