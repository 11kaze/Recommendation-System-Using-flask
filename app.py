from flask import Flask,request,render_template
import pickle
import csv
import pandas as pd
import numpy as np
from recomm_gen import *
import time


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("login.html")

with open('id_and_password.csv', mode='r') as infile:
    reader = csv.reader(infile)
    database = {rows[0]:rows[1] for rows in reader}   
database.pop('id')

@app.route('/form_login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
	    return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
            final_list = generate(int(name1))
            return render_template('home.html',name=name1 , data = final_list)

@app.route('/logout',methods=['POST','GET'])
def logout():
    return render_template('login.html')

if __name__ == '__main__':
    app.debug = True
    app.run(port=7000)