from sklearn.externals import joblib
import matplotlib.image as mimage
import numpy as np
import matplotlib.pyplot as plt
import random

from docx import Document

a = ['0', u'\u0915', u'\u0916', u'\u0917', u'\u0918', u'\u0919', u'\u091A', u'\u091B', u'\u091C', u'\u091D', u'\u091E',
     u'\u091F', u'\u0920', u'\u0921', u'\u0922', u'\u0923', u'\u0924', u'\u0925', u'\u0926', u'\u0927', u'\u0928',
     u'\u092A', u'\u092B', u'\u092C', u'\u092D', u'\u092E', u'\u092F', u'\u0930', u'\u0932', u'\u0935', u'\u0936',
     u'\u0937', u'\u0938', u'\u0939', 'ksha', 'tra', 'jya']
filename = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\First programs\hindialphabetsclass.pkl'
classifier = joblib.load(filename)
data_test = np.zeros((1, 256))
count = 0
document = Document()
while True:
    i = random.randint(1, 37)
    j = random.randint(1, 300)
    path = r'C:/Users/HRUNN/Desktop/DevanagariHandwrittenCharacterDataset/Test1/character (%d)/(%d).png' % (i, j)

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
    path2 = r'C:/Users/HRUNN/Desktop/DevanagariHandwrittenCharacterDataset/Train1/character (%d)/(1).png' % output
    im1 = mimage.imread(path2)
    plt.imshow(im1)
    plt.show()
