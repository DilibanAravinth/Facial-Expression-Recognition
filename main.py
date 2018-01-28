import cv2
from model import FacialExpressionModel
import numpy as np
import matplotlib.pyplot as plt
file=open("data.txt", "a+")


rgb = cv2.VideoCapture(0)
cap = cv2.VideoCapture('funny.mp4')
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_TRIPLEX
result = np.zeros((600,600,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

def plot_graph(angry,sad,happy,disgust,fear,surprise):
    slices = [angry,sad,happy,disgust,fear,surprise]

    activities = ['Angry','Sad','Comedy','Disgusting','Horror','Surprise']
    cols = ['r','c','g','b','y','gold']
    resultval = max(happy,sad,fear,surprise,disgust,angry)
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    if(resultval==angry):
        one = 0.1
    elif(resultval==sad):
        two = 0.1
    elif(resultval==happy):
        three = 0.1
    elif(resultval==disgust):
        four = 0.1
    elif(resultval==fear):
        five = 0.1
    elif(resultval==surprise):
        six = 0.1

    plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(one,two,three,four,five,six),
        autopct='%1.1f%%')

    plt.title('Result Graph\nIt could be')
    plt.show()

def __get_data__():

    _, frame = rgb.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, frame, gray

def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    
    happy = 0
    sad = 0
    angry = 0
    disgust = 0
    neutral = 0
    fear = 0
    surprise = 0

    while True:
        ret, fram = cap.read()  
        cv2.imshow('frame',fram)

        ix += 1
        faces, frame, gray_fr = __get_data__()

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            
            if(pred == "Happy"):
                happy += 1 
            elif(pred == "Disgust"):
                disgust += 1
            elif(pred == "Fear"):
                fear += 1
            elif(pred == "Angry"):
                angry += 1
            elif(pred == "Surprise"):
                surprise += 1
            elif(pred == "Sad"):
                sad += 1
            else :
                neutral += 1

            cv2.putText(frame, pred, (5, 50), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Neutral : " + str(neutral) , (5, 80), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Happy : " + str(happy), (5, 120), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Sad : " + str(sad), (5, 160), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Fear : " + str(fear), (5, 200), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Surprise : " + str(surprise), (5, 240), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Disgust : " + str(disgust), (5, 280), font, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Angry : " + str(angry), (5, 320), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Facial Expression Recognition by Spartans', frame)
    cv2.destroyAllWindows()
    total = happy + sad + fear + surprise + disgust + angry
    cv2.putText(result,"Neutral : "+str(int((neutral/total)*100)),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Happy : "+str(int((happy/total)*100)),(10,80), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Sad : "+str(int((sad/total)*100)),(10,110), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Fear : "+str(int((fear/total)*100)),(10,140), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Surprise : "+str(int((surprise/total)*100)),(10,170), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Disgust : "+str(int((disgust/total)*100)),(10,200), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result,"Angry : "+str(int((angry/total)*100)),(10,230), font, 1,(255,255,255),1,cv2.LINE_AA)
    resultval = max(happy,sad,fear,surprise,disgust,angry)

    cv2.imshow("Result",result)
    file.write("H"+str((happy/total)*100)+" S" + str((sad/total)*100)+" F"+ str((fear/total)*100)+" SU"+ str((surprise/total)*100)+" D"+ str((disgust/total)*100)+" A"+ str((angry/total)*100)+" T"+ str(resultval)+"\n")
    cv2.waitKey(0)
    plot_graph(angry,sad,happy,disgust,fear,surprise)

    file.close() 

if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    start_app(model)