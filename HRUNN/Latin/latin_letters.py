from sklearn.externals import joblib
import matplotlib.image as  mimage
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from skimage import util
from sklearn import metrics
from collections import Counter
from PIL import Image
from docx import Document
import string

filename = r'C:\Users\HRUNN\Desktop\DevanagariHandwrittenCharacterDataset\First programs\englishalphabetclass.pkl'
classifier = joblib.load(filename)
data_test = np.zeros((1, 196))
count = 0
document = Document()
while True:
    i = random.randint(1, 26)
    j = random.randint(1, 201)
    path = r'C:\Users\HRUNN\Desktop\New folder\characters (%d)\(%d).png' % (i, j)
    im = mimage.imread(path)
    
    for i in range(0, 26, 2):
        for j in range(0, 26, 2):
            data_test[:, count] = im[i, j] + im[i + 1, j] + im[i, j + 1] + im[i + 1, j + 1]
            count = count + 1
    count = 0
    output = classifier.predict(data_test)
    a = string.ascii_uppercase[int(output) - 1:int(output)]
    document.add_paragraph(a)
    document.save('English.docx')
    plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.subplot(2, 1, 2)
    path2 = r'C:\Users\HRUNN\Desktop\New folder\characters (%d)\(1).png' % output
    im1 = mimage.imread(path2)
    plt.imshow(im1)
    plt.show()
