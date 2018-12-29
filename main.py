from ciper_label_generator.file_parser import extract_news
from labeler.LabelFactory import LabelFactory

if __name__ == '__main__':
    data_folder_path_a = "./ciper_label_generator/dataset/"
    attributes_file_path_a = "./ciper_label_generator/dataset_ciperchile.txt"
    list_of_labels_n_text = extract_news(attributes_file_path_a, data_folder_path_a)
    x = [label_n_text[1] for label_n_text in list_of_labels_n_text]  # list of texts
    y = [label_n_text[0] for label_n_text in list_of_labels_n_text]  # list of list of labels

    labeler = LabelFactory(y)
    new_y = labeler.binary_label()
    print("done")

