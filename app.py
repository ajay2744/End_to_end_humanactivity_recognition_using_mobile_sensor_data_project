from flask import Flask,render_template,request
from src.utils import load_model
from src.pipeline.predict_pipeline import predict
import numpy as np

app=Flask(__name__)

@app.route("/")
def home():
         return render_template("input_form.html")

@app.route("/predict",methods=['GET','POST'])
def predict_activity():
       
       if request.method=='GET':
              return render_template("input_form.html")
       else:
            data=[value for value in request.form.values()]
            data=np.array([data])
            print(data)

            output=predict(data)
            if output[0]==1:
                  results="standing"
            elif output[0]==0:
                  results="walking"
            print("The predicted human activity is:{}".format(results))

            return render_template("input_form.html",results=results)

            

       
if __name__=="__main__":
    app.run(debug=True)



