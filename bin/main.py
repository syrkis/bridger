# imports
import sys
import logging
from tqdm import tqdm
from reader import parse

FORMAT = '%(name)s: %(asctime)s %(message)s'
timeformat = '%m-%d %H:%M:%S'
logging.basicConfig(format=FORMAT,
                    datefmt=timeformat,
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger('MAIN')
import sentiment

sentiment.main()