import csv
import pickle

from text_preprocess import soft_clean, full_clean


def read_attributes(attributes_file_path, label="tags"):
    """
    Parse the attribute file.
    :param attributes_file_path: Path to attribute file.
    :param label: label to be extracted from metadata.
    :return:
    """
    with open(attributes_file_path, 'r', encoding='UTF-8') as attributes_file:
        attributes = csv.DictReader(attributes_file, delimiter='\t',)
        if label is "tags":
            return [[article_attributes['filename'], article_attributes['temas'].split(';')] for article_attributes in attributes]
        elif label is "author":
            ret = []
            for article_attributes in attributes:
                aggr = []
                for substr in article_attributes['por'].split('y '):
                    for substr in substr.split(','):
                        if substr is not "":
                            substr = substr.strip()
                            substr = substr.split(" (")[0]
                            substr = substr.lower()
                            aggr.append(substr)
                ret.append([article_attributes['filename'], aggr])
            return ret


def extract_articles(attributes, data_folder_path):
    attributes_copy = attributes
    for article_attributes in attributes_copy:
        with open(data_folder_path+article_attributes[0], 'r', encoding='UTF-8') as article:
            article_attributes.append(article.read().replace("\n", " "))
    return attributes_copy


def extract_news(attributes_file_path, data_folder_path, label="tags"):
    """
    Extract news and labels
    :param attributes_file_path:
    :param data_folder_path:
    :param label: can be "tags" or "author" within the news
    :return: List of tuples which contains each list of labels and the corresponding text
    """
    print("info: loading news")
    try:
        ret = pickle.load(open("./cache/news.pickle", 'rb'))
    except IOError:
        attributes = read_attributes(attributes_file_path, label=label)
        article_data = extract_articles(attributes, data_folder_path)
        ret = [[[soft_clean(label) for label in data[1]], full_clean(data[2])] for data in article_data]
        pickle.dump(ret, open("./cache/news.pickle", 'wb'))
    return ret
