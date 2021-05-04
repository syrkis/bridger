# imports
from indexer import main
import os
data_path = '/home/common/datasets/amazon_review_data_2018/reviews'
for file_name in os.listdir(data_path):
    file_path = data_path + "/" +  file_name
    main(file_path)
