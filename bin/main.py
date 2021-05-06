# imports
import sys
import logging
import indexer

FORMAT = '%(name)s: %(asctime)s %(message)s'
timeformat = '%m-%d %H:%M:%S'
logging.basicConfig(format=FORMAT,
                    datefmt=timeformat,
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger('MAIN')
import sentiment

#sentiment.main()
indexer.main()
