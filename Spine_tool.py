
import os
import pandas as pd
import numpy as np
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
import pickle
import PySimpleGUI as sg


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    layout = [ [sg.Text('Spinal Disease', justification='center', font=("Helvetica", 25))],
          [sg.Text('Management (0-, 1-, 2-) ', size = (40,1)), sg.InputText()],
         [sg.Text('Age', size=(40,1)), sg.InputText()],
         [sg.Text('Gender (0 for Male, 1 for Female)', size=(40,1)), sg.InputText()],
         [sg.Text('BMI', size=(40,1)), sg.InputText()],
         [sg.Text('Occupation (0-sedentary, 1-light, 2-medium, 3-heavy)', size=(40,1)), sg.InputText()],
         [sg.Text('Exercise (0-No, 1-Yes)', size = (40,1)), sg.InputText()],
         [sg.Text('Ambulation (0-independent, 1-with support, 3-bedridden)', size = (40,1)), sg.InputText()],
         [sg.Text('Chief Complaint (0-Back, 1-Leg, 2-Back/Leg)', size = (40,1)), sg.InputText()],
         [sg.Text('Duration of Symptoms (0-<3months, 1->3-6months, 2->6months)', size = (40,1)), sg.InputText()],
         [sg.Text('History of past spine surgery (0-No, 1-Yes)', size = (40,1)), sg.InputText()],
         [sg.Text('Hamilton Anxiety Rating Scale ', size = (40,1)), sg.InputText()],  
         [sg.Text('Neurological Deficit ', size = (40,1)), sg.InputText()],
         [sg.Text('Listhesis ', size = (40,1)), sg.InputText()],
         [sg.Text('Disc degeneration grading on T2 ', size = (40,1)), sg.InputText()],
         [sg.Text('Modic Classification ', size = (40,1)), sg.InputText()],
         [sg.Text('Disc Herniation ', size = (40,1)), sg.InputText()],
         [sg.Text('Perineural Intraforaminal fat ', size = (40,1)), sg.InputText()],
         [sg.Text('Compression of Central Spinal Canal (0: CSA >100mm2; 1: CSA 75-100mm2; 2:CSA <75mm2)', size = (40,1)), sg.InputText()],
         [sg.Text('Hypertrophy of ligamentum Flavum (0: <2mm; 1: 2-4mm; 2: >4mm)', size = (40,1)), sg.InputText()],
#          [sg.Text('Reduction of posterior epidural fat (0: Y sign absent; 1: Y sign present)', size = (40,1)), sg.InputText()],
         [sg.Text('Nerve root compression in lateral recess ', size = (40,1)), sg.InputText()],
         [sg.Text('Hypertrophic facet joint degeneration', size = (40,1)), sg.InputText()],
         [sg.Text('Foraminal nerve root impingement (0-None, 1-Touching, 2-Displacing, 3-Compressing)', size = (40,1)), sg.InputText()],
         [sg.Text('Size and Shape of Foramenn ', size = (40,1)), sg.InputText()],
         [sg.Text('Levels Involved', size = (40,1)), sg.InputText()],
         [sg.Text('Levels Operated', size = (40,1)), sg.InputText()],
         [sg.Text('VAS before surgery', size = (40,1)), sg.InputText()],
         [sg.Text('MOD before surgery', size = (40,1)), sg.InputText()],
         [sg.Text('NCOS before surgery ', size = (40,1)), sg.InputText()],
         
         [sg.Ok('Submit'), sg.Cancel()]]

    window = sg.Window('Spinal Disease').Layout(layout)

    button, values = window.Read()
    window.close()

    vas = []
    for i in range(1,25):
        vas.append(float(values[i]))
#     print(vas)

    mod = []
    for i in range(1,25):
        mod.append(float(values[i]))

    ncos = []
    for i in range(1,25):
        ncos.append(float(values[i]))

    from sklearn.ensemble import RandomForestClassifier
    # Fitting Random Forest Classification to the Training set
    filename = 'model/VAS_withoutM.sav'
    VAS_withoutM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')

    filename = 'model/VAS_withM.sav'
    VAS_withM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')


    filename = 'model/MOD_withoutM.sav'
    MOD_withoutM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')

    filename = 'model/MOD_withM.sav'
    MOD_withM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')


    filename = 'model/NCOS_withoutM.sav'
    NCOS_withoutM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')

    filename = 'model/NCOS_withM.sav'
    NCOS_withM = pickle.load(open(filename, 'rb'))
    warnings.filterwarnings('ignore')




    if float(values[25])>=0 and float(values[25])<=2:
        vas.append(0)
    elif float(values[25])>=3 and float(values[25])<=4:
        vas.append(1)
    elif float(values[25])>=5 and float(values[25])<=6:
        vas.append(2)
    elif float(values[25])>=7 and float(values[25])<=8:
        vas.append(3)
    elif float(values[25])>=9 and float(values[25])<=10:
        vas.append(4)

    vas=np.array(vas)

    vas_new=[]

    vas_new.append(float(values[0]))
    vas_new.append(vas[24])
#     print(vas_new)
    vas_new=np.array(vas_new)
    vas=vas.reshape(1, -1)
    vas_new=vas_new.reshape(1,-1)

    if float(values[26])>=0 and float(values[26])<=10:
        mod.append(0)
    elif float(values[26])>=11 and float(values[26])<=20:
        mod.append(1)
    elif float(values[26])>=21 and float(values[26])<=30:
        mod.append(2)
    elif float(values[26])>=31 and float(values[26])<=40:
        mod.append(4)
    elif float(values[26])>=41 and float(values[26])<=50:
        mod.append(5)
    elif float(values[26])>=51 and float(values[26])<=60:
        mod.append(6)

    mod=np.array(mod)

    mod_new=[]

    mod_new.append(float(values[0]))
    mod_new.append(mod[24])
    mod_new=np.array(mod_new)
    mod=mod.reshape(1, -1)
    mod_new=mod_new.reshape(1,-1)
#     print(mod_new)

    if float(values[27])>=0 and float(values[27])<=10:
        ncos.append(0)
    elif float(values[27])>=11 and float(values[27])<=20:
        ncos.append(1)
    elif float(values[27])>=21 and float(values[27])<=30:
        ncos.append(2)
    elif float(values[27])>=31 and float(values[27])<=40:
        ncos.append(4)
    elif float(values[27])>=41 and float(values[27])<=50:
        ncos.append(5)
    elif float(values[27])>=51 and float(values[27])<=60:
        ncos.append(6)
    elif float(values[27])>=61 and float(values[27])<=70:
        ncos.append(7)
    elif float(values[27])>=71 and float(values[27])<=80:
        ncos.append(7)
    elif float(values[27])>=81 and float(values[27])<=90:
        ncos.append(8)
    elif float(values[27])>=91 and float(values[27])<=100:
        ncos.append(9)

    ncos=np.array(ncos)

    ncos_new=[]

    ncos_new.append(float(values[0]))
    ncos_new.append(ncos[24])
    ncos_new=np.array(ncos_new)
    ncos=ncos.reshape(1, -1)
    ncos_new=ncos_new.reshape(1,-1)
#     print(ncos_new)

    y_pred_vas = VAS_withoutM.predict(vas)
    y_pred2 = VAS_withM.predict(vas_new)
    x=y_pred_vas+y_pred2/2
    z=x[0]
    print('Management:', float(values[0]))
    print('predicted VAS improvement class:', round(z))

    y_pred_mod = MOD_withoutM.predict(mod)
    y_pred2 = MOD_withM.predict(mod_new)
    x=y_pred_mod+y_pred2/2
    z=x[0]

    print('predicted MOD improvement class:', round(z))

    y_pred_ncos = NCOS_withoutM.predict(mod)
    y_pred2 = NCOS_withM.predict(ncos_new)
    x=y_pred_mod+y_pred2/2
    z=x[0]

    print('predicted NCOS improvement class:', round(z))


    for i in range(0,2):
        layout2 = [ [sg.Text('Change Management', justification='center', font=("Helvetica", 25))],
                 [sg.Text('Management', size=(40,1)), sg.InputText()],
                 [sg.Ok('Submit'), sg.Cancel()]]

        window2 = sg.Window('Spinal Disease').Layout(layout2)

        button, val = window2.Read()
        window2.close()
        vas_new=[]
        vas_new.append(float(val[0]))
        vas_new.append(vas[0,24])
        vas_new=np.array(vas_new)
        vas_new=vas_new.reshape(1,-1)
        y_pred2 = VAS_withM.predict(vas_new)
#         print(vas_new)
        #print(y_pred2)
        x=y_pred_vas+y_pred2/2
        z=x[0]

        print('Management:', float(val[0]))

        print('predicted VAS improvement class:', round(z))

        mod_new=[]
        mod_new.append(float(val[0]))
        mod_new.append(mod[0,24])
        mod_new=np.array(mod_new)
        mod_new=mod_new.reshape(1,-1)
        y_pred2 = MOD_withM.predict(mod_new)
#         print(mod_new)
        x=y_pred_mod+y_pred2/2
        z=x[0]
        print('predicted MOD improvement class:', round(z))

        ncos_new=[]
        ncos_new.append(float(val[0]))
        ncos_new.append(ncos[0,24])
        ncos_new=np.array(ncos_new)
        ncos_new=ncos_new.reshape(1,-1)
        y_pred2 = NCOS_withM.predict(ncos_new)
#         print(ncos_new)
        x=y_pred_ncos+y_pred2/2
        z=x[0]
        print('predicted NCOS improvement class:', round(z))
