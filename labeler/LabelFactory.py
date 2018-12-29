class LabelFactory:
    def __init__(self, y):
        self.binary_labels = None
        self.multiclass_labels = None
        self.y = y
        metrics = {'label_freq': {

        }}
        for labels in self.y:
            for label in labels:
                try:
                    metrics['label_freq'][label] += 1
                except KeyError:
                    metrics['label_freq'][label] = 1
        self.sorted_labels = []
        for label, freq in metrics['label_freq'].items():
            if label != "":
                self.sorted_labels.append((label, freq))
        self.sorted_labels.sort(key=lambda tup: tup[1], reverse=True)

    def binary_label(self, top=0):
        positive_label = self.sorted_labels[top][0]
        self.binary_labels = []
        self.positive_count = 0
        self.negative_count = 0
        for labels in self.y:
            if positive_label in labels:
                self.binary_labels.append(1)
                self.positive_count += 1
            else:
                self.binary_labels.append(0)
                self.negative_count += 1
        print("Binary Label stats: Positives {}, Negatives {}".format(self.positive_count, self.negative_count))
        return self.binary_labels

    def top_labels(self, count=10):
        top_labels = self.sorted_labels[0:count]
        self.multiclass_labels = []
        for labels in self.y:
            parcial_labels = []
            for label in labels:
                if label in top_labels:
                    parcial_labels.append(label)
            self.multiclass_labels.append(parcial_labels)
        return self.multiclass_labels
