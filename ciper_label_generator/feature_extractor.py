from text_preprocess.compound_cleaners import soft_clean


from ciper_label_generator.file_parser import read_attributes, extract_articles, extract_labels


def normalize_labels(article_labels, major_labels):
    lel = []
    for label in article_labels:
        clean_label = soft_clean(label)
        if clean_label in major_labels:
            lel.append(clean_label)
    return lel
    #return [label for label in article_labels if label in major_labels]


def normalize_article(article):
    return soft_clean(article)


def extract_news(attributes_file_path, data_folder_path):
    attributes = read_attributes(attributes_file_path)
    article_data = extract_articles(attributes, data_folder_path)
    major_labels = [label[0] for label in extract_labels(attributes_file_path)]
    print(major_labels)
    return [(normalize_labels(data[1], major_labels), normalize_article(data[2])) for data in article_data]


if __name__ == '__main__':
    data_folder_path_a = "./dataset/"
    attributes_file_path_a = "./dataset_ciperchile.txt"
    lel = extract_news(attributes_file_path_a, data_folder_path_a)
    for a in lel:
        print(a)