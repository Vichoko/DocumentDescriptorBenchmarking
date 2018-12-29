import csv

from text_preprocess import soft_clean


def read_attributes(attributes_file_path, label="tags"):
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


def normalize_text(article):
    return soft_clean(article)


def extract_news(attributes_file_path, data_folder_path):
    attributes = read_attributes(attributes_file_path, label='tags')
    article_data = extract_articles(attributes, data_folder_path)
    return [[[normalize_text(label) for label in data[1]], normalize_text(data[2])] for data in article_data]
