from ciper_label_generator.file_parser import extract_news
from descriptor.DescriptorFactory import DescriptorFactory
from labeler.LabelFactory import LabelFactory

data_folder_path_a = "./ciper_label_generator/dataset/"
attributes_file_path_a = "./ciper_label_generator/dataset_ciperchile.txt"

if __name__ == '__main__':
    list_of_labels_n_text = extract_news(attributes_file_path_a, data_folder_path_a)
    x = [label_n_text[1] for label_n_text in list_of_labels_n_text]  # list of texts
    y = [label_n_text[0] for label_n_text in list_of_labels_n_text]  # list of list of labels

    labeler = LabelFactory(y)
    descriptor = DescriptorFactory(x)

    y = labeler.binary_label()
    new_descriptors = {
        **descriptor.fast_text(),
        **descriptor.glove(),
        **descriptor.word2vec()
    }
    print("info: loaded all vectors")

    # get_classifier_benchmarks(new_descriptors, y)  # todo: implement

    print("done")

