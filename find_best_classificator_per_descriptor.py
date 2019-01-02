from ciper_label_generator.file_parser import extract_news
from classification.Benchmark import get_classifier_benchmarks
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
    best_models = {}
    descriptor_methods = [
        # descriptor.bag_of_words,
        # descriptor.tf_idf,
        descriptor.fasttext,
        descriptor.word2vec,
        descriptor.glove,
        # descriptor.fasttext_idf,
        # descriptor.word2vec_idf,
        # descriptor.glove_idf,
    ]
    print("info: starting classificator benchmarking")
    for descriptor_method in descriptor_methods:
        for descriptor_name, x in descriptor_method().items():
            metrics = get_classifier_benchmarks(x, y, descriptor_name)

            # find best model
            best_f1 = 0
            best_clf = "Base"
            for classificator_name, clf_metrics in metrics.items():
                if clf_metrics["educacion"]['f1'] > best_f1:
                    best_f1 = clf_metrics["educacion"]['f1']
                    best_clf = classificator_name
            best_models[descriptor_name] = metrics[best_clf]
            best_models[descriptor_name]['best clf'] = best_clf
            print(best_models[descriptor_name])
    print(best_models)
    print("done")
