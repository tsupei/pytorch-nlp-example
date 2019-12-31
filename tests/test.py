import csv
from classifier.trainee import Trainee
from classifier.data import TextData
from classifier import preprocess


def read_raw(filename):
    raw_data = []
    with open(filename, 'r', encoding="utf8") as file:
        rows = list(csv.reader(file, delimiter=","))
        for row in rows:
            raw_data.append((row[0], row[1]))
    return raw_data


if __name__ == "__main__":
    csv_filename = "../data/ptt-classification-corpus.csv"
    raw_data = read_raw(csv_filename)
    data = []
    for title, label in raw_data:
        token = preprocess.convert_char_token(title)
        data.append((token, label))
    train_data = TextData(doc_size=50)
    train_data.process(data)

    trainee = Trainee()
    trainee.train(train_data=train_data, epoch=3, bsz=4, lr=0.001)

