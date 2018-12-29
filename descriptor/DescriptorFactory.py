import numpy
from gensim.models import KeyedVectors

fasttext_wordvector_file = './descriptor/vectors/fasttext-sbwc.3.6.e20.vec'
glove_wordvector_file = './descriptor/vectors/glove-sbwc.i25.vec'
wor2vec_wordvector_file = './descriptor/vectors/SBW-vectors-300-min5.txt'


class DescriptorFactory:
    def __init__(self, x):
        """
        Init descriptor facotry

        :param x: List of documents (strings)
        """
        self.x = x

    def bag_of_words(self):
        """
        Return BoW descriptors
        :return: List of bag-of-words
        """
        #todo: implement
        return

    def load_vectors(self, descriptor_name: str, vector_filepath: str):
        """
        Load vectors to KeyedVectors object (dict-like)
        :param descriptor_name: Name of the descriptor
        :param vector_filepath: Filepath to vector file in word2vec format (.vec)
        :return: 
        """
        print("info: starting to load {} vectors (2.2 GB) (~4 minutes)".format(descriptor_name))
        try:
            wordvectors = KeyedVectors.load_word2vec_format(vector_filepath)
            print("info: done loading")
            return wordvectors
        except IOError as e:
            print("error: exception raised while loading {}".format(descriptor_name))
            print(str(e))
            print("warning: skipping {} descriptor".format(descriptor_name))
            return None

    def texts_to_vectors(self, wordvectors: KeyedVectors, descriptor_name: str):
        """
        Map each document's word to a vector contained in wordvectors
        and then calculate a sentence vector by averaging all the vectors of the document.
        :param wordvectors: KeyedVector object with the loaded vectors
        :param descriptor_name: Name of the descriptor
        :return:
        """
        new_x = []
        vectorized_counter = 0
        not_vectorized_counter = 0
        for document in self.x:
            document_vectors = []
            for word in document.split(" "):
                try:
                    document_vectors.append(wordvectors.get_vector(word))
                    vectorized_counter += 1
                except KeyError:
                    # print("warning: word: \"{}\" not found in {} vectors".format(word, descriptor_name))
                    not_vectorized_counter += 1
                    continue
            new_x.append(numpy.mean(document_vectors, axis=0))  # todo: use TF-IDF instead of simple mean and compare
        print("info: done converting. vectorized {}; skipped {}".format(vectorized_counter, not_vectorized_counter))
        return new_x

    def fasttext(self):
        """
        Retrun fastText descriptors.
        fastText vectors are loaded and then freed to optimize memory.
        :return: List of vectors
        """
        descriptor_name = "FastText"
        vector_file = fasttext_wordvector_file
        wordvectors = self.load_vectors(descriptor_name, vector_file)
        print("info: starting converting texts")
        if wordvectors:
            return {descriptor_name: self.texts_to_vectors(wordvectors, descriptor_name)}
        return {}

    def word2vec(self):
        """
        Return word2vec skip-gram vectors
        :return: List of vectors
        """
        descriptor_name = "Word2Vec"
        vector_file = wor2vec_wordvector_file
        wordvectors = self.load_vectors(descriptor_name, vector_file)
        print("info: starting converting texts")
        if wordvectors:
            return {descriptor_name: self.texts_to_vectors(wordvectors, descriptor_name)}
        return {}

    def glove(self):
        """
        Return glove vectors
        :return:
        """
        descriptor_name = "Glove"
        vector_file = glove_wordvector_file
        wordvectors = self.load_vectors(descriptor_name, vector_file)
        print("info: starting converting texts")
        if wordvectors:
            return {descriptor_name: self.texts_to_vectors(wordvectors, descriptor_name)}
        return {}


