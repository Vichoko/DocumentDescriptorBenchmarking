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
        return

    def fast_text(self):
        """
        Retrun fastText descriptors
        :return: List of vectors
        """
        return

    def word_2_vec(self):
        """
        Return word2vec skip-gram vectors
        :return: List of vectors
        """
        return

    def glove(self):
        """
        Return glove vectors
        :return:
        """
        return