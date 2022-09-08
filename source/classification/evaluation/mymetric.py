from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def confusion(y_true, y_pred, classes, path):

    cnf_matrix = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cnf_matrix,
        display_labels=classes)

    disp.plot(
        include_values=True,
        cmap='viridis_r',
        ax=ax,
        xticks_rotation='vertical')

    plt.savefig(path)


def report(y_true, y_pred, classes, path):
    # path_rp = '../report/reportFT_ResNet50_152.txt'
    path_rp = path
    try:
        s = classification_report(y_true, y_pred, target_names=classes)

        with open(path_rp, mode=r'w+') as f:
            f.write(s)

        with open(path_rp) as f:
            print(f.read())

        f.close()
    except FileExistsError:
        pass
