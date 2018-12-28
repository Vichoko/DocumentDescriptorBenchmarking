import csv
from text_preprocess.compound_cleaners import soft_clean


def read_attributes(attributes_file_path):
    with open(attributes_file_path, 'r', encoding='UTF-8') as attributes_file:
        attributes = csv.DictReader(attributes_file, delimiter='\t',)
        return [[article_attributes['filename'], article_attributes['temas'].split(';')] for article_attributes in attributes]


def extract_articles(attributes, data_folder_path):
    attributes_copy = attributes
    for article_attributes in attributes_copy:
        with open(data_folder_path+article_attributes[0], 'r', encoding='UTF-8') as article:
            article_attributes.append(article.read())
    return attributes_copy


def extract_labels(attributes_file_path):
    labels_dic = {}
    with open(attributes_file_path, 'r', encoding='UTF-8') as attributes_file:
        attributes = csv.DictReader(attributes_file, delimiter='\t', )
        for article_attributes in attributes:
            labels = [soft_clean(label) for label in article_attributes['temas'].split(';')]
            for label in labels:
                if label not in labels_dic:
                    labels_dic[label] = 1
                else:
                    labels_dic[label] += 1
    return get_top_labels(labels_dic)


def get_top_labels(labels_dic):
    labels_list = []
    for key in labels_dic:
        labels_list.append([key,labels_dic[key]])
    labels_list.sort(key=lambda x: x[1])
    labels_list.reverse()
    return labels_list

