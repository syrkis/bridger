# imports
import sys
import logging
import runner
import sentiment


FORMAT = '%(name)s: %(asctime)s %(message)s'
timeformat = '%m-%d %H:%M:%S'
logging.basicConfig(format=FORMAT,
                    datefmt=timeformat,
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger('MAIN')
sentiment.main()
