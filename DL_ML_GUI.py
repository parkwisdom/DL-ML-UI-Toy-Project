from  tkinter import *
from  tkinter.simpledialog import *
from tkinter.filedialog import *
import csv
import json
import os
import os.path
import sqlite3
import glob
import numpy as np
import tensorflow as tf

def drawSheet(cList) :
    global cellList
    if cellList == None or cellList == [] :
        pass
    else :
        for row in cellList:
            for col in row:
                col.destroy()
    rowNum = len(cList)
    colNum = len(cList[0])
    cellList = []
    # 빈 시트 만들기
    for i in range(0, rowNum):
        tmpList = []
        for k in range(0, colNum):
            ent = Entry(window, text='')
            tmpList.append(ent)
            ent.grid(row=i, column=k)
        cellList.append(tmpList)
    # 시트에 리스트값 채우기. (= 각 엔트리에 값 넣기)
    for i in range(0, rowNum):
        for k in range(0, colNum):
            cellList[i][k].insert(0, cList[i][k])

def openCSV() :
    global  csvList, input_file
    csvList = []
    input_file = askopenfilename(parent=window,
                filetypes=(("CSV파일", "*.csv"),
                           ("모든파일", "*.*")))
    filereader = open(input_file, 'r', newline='')
    csvReader = csv.reader(filereader) # CSV 전용으로 열기
    header_list = next(csvReader)
    csvList.append(header_list)
    for row_list in csvReader:  # 모든행은 row에 넣고 돌리기.
        csvList.append(row_list)
    drawSheet(csvList)
    filereader.close()

def  saveCSV() :
    global csvList, input_file
    if csvList == [] :
        return
    saveFp = asksaveasfile(parent=window, mode='w', defaultextension='.csv',
               filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    filewriter = open(saveFp.name, 'w', newline='')
    csvWrite = csv.writer(filewriter)
    for  row_list  in  csvList :
        csvWrite.writerow(row_list)
    filewriter.close()


def diabetes() :   ##########당뇨병 훈련&테스트############

    # 매개변수
    # learning_rate버튼
    learning_rate = askfloat('Learning Rate', 'Learning Rate를 입력하시오(1e-8~0.1) ', minvalue=1e-8,maxvalue=0.1)
    label0 = Label(window, text=" Learning Rate : ")
    label0.grid(row=1, column=1)
    label0val = Label(window, text=learning_rate, relief='ridge')
    label0val.grid(row=1, column=2)
    # numberofTrain버튼
    numberofTrain = askinteger('훈련 횟수', '훈련 횟수를 적어주세요(1~20001)', minvalue=1, maxvalue=20001)
    label0 = Label(window, text=" Training times: ")
    label0.grid(row=1, column=3)
    label0val = Label(window, text=numberofTrain, relief='ridge')
    label0val.grid(row=1, column=4)

    ######main
    diabetes = np.loadtxt("diabetes.csv", delimiter=',')
    # print(diabetes.shape) #(759, 9)
    # trainingData
    xdata = diabetes[:500, 0:-1]
    ydata = diabetes[:500, [-1]]

    # testData
    xtest = diabetes[501:, 0:-1]
    ytest = diabetes[501:, [-1]]

    x = tf.placeholder(tf.float32, shape=[None, 8])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    w = tf.Variable(tf.random_normal([8, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hf = tf.sigmoid(tf.matmul(x, w) + b)

    cost = -tf.reduce_mean(y * tf.log(hf) + (1 - y) * tf.log(1 - hf))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.cast(hf > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(numberofTrain):
            cv, _ = sess.run([cost, train], feed_dict={x: xdata, y: ydata})
            # if step%1000==0:
            #     print(step,cv)

        hft, pt, at = sess.run([hf, predicted, accuracy], feed_dict={x: xdata, y: ydata})
        # print("\n훈련예측값:",hft,"\n훈련예측값(0/1):",pt,"\n훈련정확도:",at)
        print("\n훈련정확도:", at)

        hfv, pv, av = sess.run([hf, predicted, accuracy], feed_dict={x:xtest,y:ytest})
        print("\n예측값:", hfv, "\n예측값(0/1):", pv, "\n정확도:", av)
        print("\ntest정확도:", av)
    ######################################################################3
        label1 = Label(window, text=" 훈련정확도 : ")
        label1.grid(row=2,column=1)
        label1val = Label(window,text=at, relief='ridge')
        label1val.grid(row=2,column=2)

        label2 = Label(window, text=" test 정확도 : ")
        label2.grid(row=3,column=1)
        label2val = Label(window,text=av, relief='ridge')
        label2val.grid(row=3,column=2)


def zoo(): ## 동물 판별기

    # 매개변수
    # learning_rate버튼
    learning_rate = askfloat('Learning Rate', 'Learning Rate를 입력하시오(1e-8~0.1) ', minvalue=1e-8, maxvalue=0.1)
    label0 = Label(window, text=" Learning Rate : ")
    label0.grid(row=1, column=1)
    label0val = Label(window, text=learning_rate, relief='ridge')
    label0val.grid(row=1, column=2)
    # numberofTrain버튼
    numberofTrain = askinteger('훈련 횟수', '훈련 횟수를 적어주세요(1~20001)', minvalue=1, maxvalue=20001)
    label0 = Label(window, text=" Training times: ")
    label0.grid(row=2, column=1)
    label0val = Label(window, text=numberofTrain, relief='ridge')
    label0val.grid(row=2, column=2)

    ######main
    xy = np.loadtxt('zoo.csv', delimiter=",", dtype=np.float32)
    xdata = xy[:, 0:-1]
    ydata = xy[:, [-1]]
    # print(xdata.shape,ydata.shape)
    nb_classes = 7  # 0~6
    x = tf.placeholder(tf.float32, [None, 16])
    y = tf.placeholder(tf.int32, [None, 1])

    # y에는 0~6사이의 임의의 수 저장
    # 원핫 인코딩 해야함.
    y_one_hot = tf.one_hot(y, nb_classes)
    print("one hot 상태: ", y_one_hot)
    # 0 -> 1000000, 3 -> 0001000
    # 원핫 인코딩을 수행하면 차원이 1 증가
    # 예를 들어 y 가 (None,1) ->(None,1,7)이 됨
    # [[0],[3]]->[[[1000000]],[[0001000]]]
    y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])  # --> -1은 전체 데이터를 정의
    print("reshape 결과: ", y_one_hot)

    w = tf.Variable(tf.random_normal([16, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(x, w) + b  # logit=score
    hf = tf.nn.softmax(logits)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)

    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    prediction = tf.argmax(hf, axis=1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(numberofTrain):
            sess.run(optimizer, feed_dict={x: xdata, y: ydata})
            if step % 100 == 0:
                cv, av = sess.run([cost, accuracy], feed_dict={x: xdata, y: ydata})
                print(step, cv, av)
    #########################################################################

        ### UI구성
        pred1= askinteger('예측해보기', 'hair(0 / 1)')
        pred2= askinteger('예측해보기', 'feathers(0 / 1)')
        pred3= askinteger('예측해보기', 'eggs(0 / 1)')
        pred4= askinteger('예측해보기', 'milk(0 / 1)')
        pred5= askinteger('예측해보기', 'airborne(0 / 1)')
        pred6= askinteger('예측해보기', 'aquatic(0 / 1)')
        pred7= askinteger('예측해보기', 'predator(0 / 1)')
        pred8= askinteger('예측해보기', 'toothed(0 / 1)')
        pred9= askinteger('예측해보기', 'backbone(0 / 1)')
        pred10= askinteger('예측해보기', 'breathes(0 / 1)')
        pred11= askinteger('예측해보기', 'venomous(0 / 1)')
        pred12= askinteger('예측해보기', 'fins(0 / 1)')
        pred13= askinteger('예측해보기', 'legs(0, 2, 4, 5, 6, 8))')
        pred14= askinteger('예측해보기', 'tail(0 / 1)')
        pred15= askinteger('예측해보기', 'domestic(0 / 1)')
        pred16= askinteger('예측해보기', 'catsize(0 / 1)')

        pred=sess.run(prediction, feed_dict={x: [[pred1,pred2,pred3,pred4,pred5,pred6
                                                  ,pred7,pred8,pred9,pred10,pred11,
                                                  pred12,pred13,pred14,pred15,pred16]]})

        label1 = Label(window, text=" 예측 원하는 데이터 : ")
        label1.grid(row=3,column=1)
        label1val = Label(window,text=[pred1,pred2,pred3,pred4,pred5,pred6
                                                  ,pred7,pred8,pred9,pred10,pred11,
                                                  pred12,pred13,pred14,pred15,pred16], relief='ridge')
        label1val.grid(row=3,column=2)

        label2 = Label(window, text=" 예상되는 동물 번호 : ")
        label2.grid(row=4,column=1)
        label2val = Label(window,text=pred, relief='ridge')
        label2val.grid(row=4,column=2)

        label3 = Label(window, text=" Set of animals : ")
        label3.grid(row=5,column=1)
        label3=Label(window,text=" 1 (41) aardvark, antelope, bear, boar, buffalo, calf,cavy, \n"
                                 "cheetah, deer, dolphin, elephant,fruitbat, giraffe, girl, goat,\n "
                                 "gorilla, hamster,hare, leopard, lion, lynx, mink, mole, mongoose,\n"
                                 "opossum, oryx, platypus, polecat, pony,porpoise, puma, pussycat, \n"
                                 "raccoon, reindeer,seal, sealion, squirrel, vampire, vole, wallaby,wolf\n"
                                 "2 (20) chicken, crow, dove, duck, flamingo, gull, hawk,kiwi, lark, ostrich,\n"
                                 "parakeet, penguin, pheasant,rhea, skimmer, skua, sparrow, swan, vulture, wren\n"
                                 "3 (5)  pitviper, seasnake, slowworm, tortoise, tuatara \n"
                                 "4 (13) bass, carp, catfish, chub, dogfish, haddock,herring, pike, piranha, seahorse, sole, stingray, tuna\n"
                                 "5 (4)  frog, frog, newt, toad\n"
                                 "6 (8)  flea, gnat, honeybee, housefly, ladybird, moth, termite, wasp\n"
                                 "7 (10) clam, crab, crayfish, lobster, octopus,scorpion, seawasp, slug, starfish, worm\n")
        label3.grid(row=5,column=2)


def iris(): #붓꽃 판별
    # learning_rate버튼
    learning_rate = askfloat('Learning Rate', 'Learning Rate를 입력하시오(1e-8~0.1) ', minvalue=1e-8, maxvalue=0.1)
    label0 = Label(window, text=" Learning Rate : ")
    label0.grid(row=1, column=1)
    label0val = Label(window, text=learning_rate, relief='ridge')
    label0val.grid(row=1, column=2)
    # numberofTrain버튼
    numberofTrain = askinteger('훈련 횟수', '훈련 횟수를 적어주세요(1~20001)', minvalue=1, maxvalue=20001)
    label0 = Label(window, text=" Training times: ")
    label0.grid(row=2, column=1)
    label0val = Label(window, text=numberofTrain, relief='ridge')
    label0val.grid(row=2, column=2)

    #main
    read = open("iris.csv", "r", encoding="utf-8")
    # print(read)
    csvread = csv.reader(read)
    # print(csvread) #<_csv.reader object at 0x0000014153D708D0>
    next(csvread)  # 첫번째 줄 skip(한줄씩 skip함)

    xdata = []
    ydata = []

    index = ['setosa', 'versicolor', 'virginica']
    for row in csvread:
        # print(row)
        data = []
        sepal_length = float(row[1])
        sepal_width = float(row[2])
        sepal_length = float(row[3])
        sepal_width = float(row[4])
        data = [sepal_length, sepal_width, sepal_length, sepal_width]
        xdata.append(data)
        # print(xdata)

        for i in range(3):
            if row[5] == index[i]:
                ydata.append([i])

    # print(xdata)
    # print(ydata)

    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.int32, shape=[None, 1])
    w = tf.Variable(tf.random_normal([4, 3]))
    b = tf.Variable(tf.random_normal([3]))

    nb_classes = 3
    y_one_hot = tf.one_hot(y, nb_classes)
    y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])
    # [[0],[2]] -one hot-> [[[100]],[[001]]] -reshape->[[100],[001]]

    logit = tf.matmul(x, w) + b
    hf = tf.nn.softmax(logit)

    costi = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_one_hot)
    cost = tf.reduce_mean(costi)

    prediction = tf.argmax(hf, 1)
    corrent_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_prediction, dtype=tf.float32))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    ## ▲그래프 정의---------------------------------

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(numberofTrain):
        cv, av, _ = sess.run([cost, accuracy, train], feed_dict={x: xdata, y: ydata})
        if step % 20 == 0:
            print('cost:', cv, 'acc:', av)

    pred = sess.run(prediction, feed_dict={x: xdata})
    ydata = np.array(ydata, dtype=np.int32)
    for p, y in zip(pred, ydata.flatten()):  # flatten 함수는 np에 있음. ydata 의 구조를 리스트->array 로 변경
        print("{} prediction: {} True Y: {}".format(p == y, p, y))



## 전역 변수 ##
csvList, cellList = [], []
input_file = ''

## 메인 코드 ##
window = Tk()
window.geometry('500x500')

mainMenu = Menu(window)
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='CSV 열기', command=openCSV)
fileMenu.add_command(label='CSV 저장', command=saveCSV)
fileMenu.add_separator()

DLdiabetesMenu = Menu(mainMenu)
mainMenu.add_cascade(label='Diabetes', menu=DLdiabetesMenu)
DLdiabetesMenu.add_command(label='Training', command=diabetes)

DLzooMenu = Menu(mainMenu)
mainMenu.add_cascade(label='Zoo', menu=DLzooMenu)
DLzooMenu.add_command(label='Training', command=zoo)
# DLzooMenu.add_command(label='Test', command=zoo(zooTest))


DLirisMenu = Menu(mainMenu)
mainMenu.add_cascade(label='iris', menu=DLzooMenu)
# DLzooMenu.add_command(label='Training', command=zooTest(zoo))

window.mainloop()