import numpy as np
import pickle
import glob
import cv2


train_data = []
label_data = []

for x in glob.glob("E:\\pre-trained\\dataset-tempe\\tempe-busuk\\*.jpg"):
    img = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    train_data.append((resized))
    label_data.append(0)

for x in glob.glob("E:\\pre-trained\\dataset-tempe\\tempe-matang\\*.jpg"):
    img = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    train_data.append((resized))
    label_data.append(1)

for x in glob.glob("E:\\pre-trained\\dataset-tempe\\tempe-mentah\\*.jpg"):
    img = cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    train_data.append((resized))
    label_data.append(2)

data_tempe = {"img_data" : np.array(train_data), "label_data" : np.array(label_data)}
print(data_tempe["img_data"].shape)
output = open('E:\\pre-trained\\DatasetTempe.p','wb')
pickle.dump(data_tempe,output)
output.close()

csvWrite = open('E:\\pre-trained\\DatasetTempe.csv','w')
columnTitleRow = "ClassID,Class\n"
csvWrite.write(columnTitleRow)
csvWrite.write("0,tempe-busuk\n")
csvWrite.write("1,tempe-matang\n")
csvWrite.write("2,tempe-mentah\n")
csvWrite.close()
