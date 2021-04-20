# imports
import reader
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def tokenize(target):
        X = []
        y = []
        print('hek')
        for review in reader.parse(target):
            if 'reviewText' in review.keys():
                X.append(word_tokenize(review['reviewText']))
            else:
                X.append("<U>")
            label = review['overall'] > 3
            y.append(label)

        return X, y


if __name__ == '__main__':
    print('EMPTY MAIN')