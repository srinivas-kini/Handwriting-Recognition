from sklearn.externals import joblib
import matplotlib.image as mimage
import numpy as np
import matplotlib.pyplot as plt

from docx import Document

filename = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\First programs\englishdigitclass.pkl'
classifier = joblib.load(filename)
data_test = np.zeros((1, 196))
count = 0
document = Document()
while True:
    path = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\1.png'
    im = mimage.imread(path)
    im = im[:, :, 0]
    plt.imshow(im)
    plt.show()
    for i in range(0, 26, 2):
        for j in range(0, 26, 2):
            data_test[0, count] = im[i, j] + im[i + 1, j] + im[i, j + 1] + im[i + 1, j + 1]
            count = count + 1
    count = 0
    output = classifier.predict(data_test)
    document.add_paragraph(str(output))
    document.save('English.docx')
    plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.subplot(2, 1, 2)
    path2 = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\training\%d\(1).png' % output
    im1 = mimage.imread(path2)
    plt.imshow(im1)
    plt.show()
