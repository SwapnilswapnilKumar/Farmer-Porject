from flask import Flask,render_template, request,url_for
import numpy as np
import  joblib
import pandas as pd

std = joblib.load('./models/standard_scalar.lb')
model = joblib.load('./models/kmean_model.lb')

app = Flask(__name__)

@app.route('/')
def home():
   return  render_template('home.html')

@app.route('/project',methods=["POST","GET"])
def project():
    return render_template('project.html')

@app.route('/predict',methods=["POST","GET"])
def prediction():
    if request.method =='POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        data_li=[N,P,K,temp,humidity,ph,rainfall]
        data_array=np.array(data_li).reshape(1,-1)
        data_converted = std.transform(data_array)
        cluster = model.predict(data_converted)
        
        # print(cluster)
        all_list=[]
        ref = cluster[0]
        def fun(ref):
            csv_data = pd.read_csv('./models/app_data.csv')
            
            for i in range(0,8):
                all_list.append(list(csv_data[csv_data['cluster']==i].value_counts().index))

            ans_list=[]
            temp = all_list[ref]
            for j in range(len(all_list[ref])):
                ans_list.append(all_list[ref][j][1])

            comma_separated_string = ', '.join(ans_list)

            return comma_separated_string
            

      
        return render_template('final.html',output=fun(ref))


if __name__ == "__main__":
    app.run(debug=True)