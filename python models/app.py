from flask import Flask, jsonify , render_template , request 
from waitress import serve
import pickle
import numpy as np

model=pickle.load(open('ml.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('planner.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    global final
    final=[np.array(int_features)]
    

    # prediction=model.predict_proba(final)
    # output=
# n=data1print(final)
predict()
print(final)
#flask hosting

if __name__=='__main':
    app.run

'''
mode='prod'
if __name__ == '__main__':
    if mode=='prod':
        app.run(host='0.0.0.0' , port=5500 , debug=True)
    else:
         serve(app, host='0.0.0.0', port=5500, threads=10 )  
'''
