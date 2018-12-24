import unittest

from text_preprocess import remove_stopwords, soft_clean, lemmatize


class TestPreprocess(unittest.TestCase):
    def test_stopword(self):
        for text in [
            "me gustan los gatos",
            "ayer fui a comer comida, estaba muy buena",
            "una vez el presidente se sentó con el ministro no hubo otro desenlace",
        ]:
            cleansed = remove_stopwords(text)
            self.assertGreater(len(text), len(cleansed))

    def test_normalize(self):
        for text in [
            "a veces hay muchos simbolos!",
            " otras, se separa por otros",
            " puede ser que no   hayan simbolos pero espacion mal hechos ",
        ]:
            cleansed = soft_clean(text)
            self.assertGreater(len(text), len(cleansed))

    def test_lemmatizer(self):
        for text in [
            "queria estar corriendo",
        ]:
            cleansed = lemmatize(text)
            self.assertGreater(len(text), len(cleansed))
