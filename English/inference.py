# Entry point for English Inference
import sys
import os
sys.path.append('/content/project')
from fastspeech2.infer import main

if __name__ == '__main__':
    main(config_path='/content/project/English/config.yaml')
