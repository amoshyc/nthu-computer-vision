import io
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from flask import Flask, request

app = Flask(__name__)

FIELDS = ['BR_x', 'BR_y', 'BL_x', 'BL_y', 'TL_x', 'TL_y', 'TR_x', 'TR_y']
df_true = pd.read_csv('./assets/hw3/test.csv')
y_true = df_true[FIELDS].values.reshape(-1, 4, 2)


@app.route('/cs6550', methods=['POST'])
def metric():
    file = request.files.get('file', None)
    if not file:
        return 'Invalid'

    data = io.BytesIO()
    file.save(data)
    data = data.getvalue().decode('utf-8')
    data = io.StringIO(data)
    df_pred = pd.read_csv(data)
    if not (df_pred.name == df_true.name).all():
        return 'Invalid: Name'
    if not all(is_float_dtype(df_pred[field]) for field in FIELDS):
        return 'Invalid: Expect data to be float'

    y_pred = df_pred[FIELDS].values.reshape(-1, 4, 2)
    rmse = np.sqrt(((y_true - y_pred) ** 2).sum(axis=2).mean()).item()

    print(rmse)

    return {
        'rmse': round(rmse, 3),
    }

    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # curl -F "file=@sample.csv" -X POST 140.114.76.113:5000/cs6550 -i
