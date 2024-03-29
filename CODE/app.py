import pandas as pd
from flask import Flask,render_template,request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.cluster import KMeans
import mysql
from mysql.connector import cursor



mydb = mysql.connector.connect(host='localhost', user='root', password='', port='3306', database='diabetes')
app = Flask(__name__)


df = pd.read_csv(r'pima-data.csv')

encoder = LabelEncoder()
df['diabetes'] = encoder.fit_transform(df['diabetes'])

df['bmi'] = df['bmi'].apply(np.int64)
df['diab_pred'] = df['diab_pred'].apply(np.int64)
df['skin'] = df['skin'].apply(np.int64)


global  x_test, x_train, y_test, y_train
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

pca = PCA(n_components=8)
pca.fit(x)
x_pca = pca.transform(x)

m = x_pca
n = y

m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.33, random_state=32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=32)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        sql = "SELECT * FROM dc WHERE Email=%s and Password=%s"
        val = (email, password)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loginhomepage.html', msg='success')
        else:
            return render_template('login.html', msg='fail')
    return render_template('login.html')

@app.route("/Register", methods=['GET', 'POST'])
def Register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        psw = request.form['psw']
        cpsw = request.form['cpsw']
        if psw == cpsw:
            sql = 'SELECT * FROM dc'
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('Register.html', msg='exists')
            else:
                sql = 'INSERT INTO dc(name,Email,Password) values(%s,%s,%s)'
                cur = mydb.cursor()
                values = (name,email, psw)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('Register.html', msg='Success')
        else:
            return render_template('Register.html', msg='Mismatch')
    return render_template('Register.html')

@app.route("/uploaddata", methods=['GET', 'POST'])
def uploaddata():
    if request.method == "POST":
        file = request.files['file']
        print(file)
        global df
        df = pd.read_csv(file)
        print(df)


        return render_template('uploaddata.html',msg='Success')
    return render_template('uploaddata.html')


@app.route("/viewdata")
def viewdata():
    print(df)
    a = df
    print(a)

    return render_template('viewdata.html', cols=a.columns.values, rows=a.values.tolist())
    return render_template('viewdata.html')

@app.route("/loginhomepage")
def loginhomepage():
    return render_template('loginhomepage.html')




@app.route("/training", methods=['GET', 'POST'])
def training():
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(x.shape)

    pca = PCA(n_components=9)
    pca.fit(x)
    x_pca = pca.transform(x)

    m = x_pca
    n = y

    m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.33, random_state=32)



    if request.method == "POST":

        model = request.form['algo']

        if model == "1":
            print('kkkkkkkk')
            lda = LinearDiscriminantAnalysis()
            lda.fit_transform(m, n)
            y_preds = lda.predict(m_test)
            z=accuracy_score(n_test,y_preds)*100
            print(z)
            ldadc = 'Accuracy of  LinearDiscriminantAnalysis :' + str(z)
            return render_template('training.html', msg=ldadc)
        elif model == "2":
            print('eeeeeee')
            km = KMeans(n_clusters=2, random_state=42)
            km.fit(m)
            kmc = km.predict(m)
            print('qqqqqq')
            score = silhouette_score(m, km.labels_, metric='euclidean')*100
            print(score)
            kmdc = 'Accuracy of KMeans  :  ' + str(score)

            return render_template('training.html', msg=kmdc)
        elif model=='3':
            clf = svm.SVC()
            print('dddddd')
            clf.fit(m_train, n_train)
            y_pred = clf.predict(m_test)
            m=accuracy_score(n_test, y_pred) * 100
            print(m)
            svcdc = 'Accuracy of SVM :  ' + str(m)
            return render_template('training.html', msg=svcdc)
        elif model=='4':
            xgb = XGBClassifier(random_state=10)
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)
            xgbc = accuracy_score(y_test, y_pred)*100
            print(xgbc)
            xgbdc = 'Accuracy of XGBClassifier :  ' + str(xgbc)
            return render_template('training.html', msg=xgbdc)
        elif model=='5':
            dt = DecisionTreeClassifier(random_state=10)
            dt.fit(x_train, y_train)
            xyz = dt.predict(x_test)
            dtc = accuracy_score(y_test, xyz)*100
            print(dtc)
            dtdc = 'Accuracy of DecisionTreeClassifier :  ' + str(dtc)
            return render_template('training.html', msg=dtdc)
        elif model == '6':
            ab = AdaBoostClassifier(random_state=10)
            ab.fit(x_train, y_train)
            zxc = ab.predict(x_test)
            abc = accuracy_score(y_test, zxc)*100
            print(abc)
            abdc = 'Accuracy of AdaBoostClassifier :  ' + str(abc)
            return render_template('training.html', msg=abdc)

        else :
            return render_template('training.html', msg="Please select a model")
    return render_template('training.html')



@app.route('/detection', methods=['GET','POST'])
def detection():
    if request.method == "POST":
        numpreg=request.form['numpreg']
        print(numpreg)
        glucoseconc = request.form["glucoseconc"]
        print(glucoseconc)
        diastolicbp = request.form["diastolicbp"]
        print(diastolicbp)
        thickness = request.form["thickness"]
        print(thickness)
        insulin = request.form["insulin"]
        print(insulin)
        bmi = request.form["bmi"]
        print(bmi)
        diabpred = request.form["diabpred"]
        print(diabpred)
        age = request.form["age"]
        print(age)
        skin = request.form["skin"]
        print(skin)
        mna=[numpreg, glucoseconc, diastolicbp, thickness, insulin, bmi, diabpred, age, skin]
        model = AdaBoostClassifier(random_state=10)
        model.fit(x_train, y_train)
        output=model.predict([mna])
        print(output)
        if output==0:
            msg = '<span style = color:black;>The Patient is <span style = color:red;>not a Diabetic</span></span>'
        else:
            msg = '<span style = color:black;>The Patient <span style = color:red;>has Diabetes</span></span>'


        return render_template('detection.html',msg=msg)

    return render_template('detection.html')



if __name__ == '__main__':
    app.run(debug=True)

