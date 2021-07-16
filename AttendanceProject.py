import cv2
import numpy as np
import face_recognition
import  os
from datetime import  datetime
import  cvzone
path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('AttendanceMark.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
#

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x1, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x1,y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
            markAttendance(name)

        else:
            name = "Unknown Person".upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x1, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('WebCam', img)
    cv2.waitKey(1)




# imgElon = face_recognition.load_image_file('ImageBasics/ElonMusk.png')
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('ImageBasics/ElonMuskTest.png')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#
#
# faceloc = face_recognition.face_locations(imgElon)[0]
#
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]),(255, 0, 255), 2)
#
# faceloctest = face_recognition.face_locations(imgTest)[0]
# encodeElontest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]),(255, 0, 25),2)
#
# results = face_recognition.compare_faces([encodeElon], encodeElontest)
# faceDis = face_recognition.face_distance([encodeElon],encodeElontest)