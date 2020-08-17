from sklearn.externals import joblib
import matplotlib.image as mimage
import numpy as np
import matplotlib.pyplot as plt
import random

from docx import Document

filename = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\First programs\hindidigitclass.pkl'
classifier = joblib.load(filename)
a = [u'\u0966', u'\u0967', u'\u0968', u'\u0969', u'\u096A', u'\u096B', u'\u096C', u'\u096D', u'\u096E', u'\u096F']
data_test = np.zeros((1, 256))
count = 0
document = Document()
while True:
    k = random.randint(0, 9)
    l = random.randint(1, 300)
    path = r'C:/Users/HRUNN/Desktop/DevanagariHandwrittenCharacterDataset/Test1/digit_%d/(%d).png' % (k, l)
    im = mimage.imread(path)

    for i in range(0, 30, 2):
        for j in range(0, 30, 2):
            data_test[0, count] = im[i, j] + im[i + 1, j] + im[i, j + 1] + im[i + 1, j + 1]
            count = count + 1
    count = 0
    output = classifier.predict(data_test)
    document.add_paragraph(a[int(output)])
    document.save('Hindi.docx')
    plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.subplot(2, 1, 2)
    path2 = r'C:/Users/HRUNN/Desktop/DevanagariHandwrittenCharacterDataset/Train1/digit_%d/(30).png' % output
    im1 = mimage.imread(path2)
    plt.imshow(im1)
    plt.show()
