import pathlib
import importlib.util
import pandas as pd

root = pathlib.Path(__file__).resolve().parent
app_path = root / 'Random forest model' / 'streamlit_app.py'
spec = importlib.util.spec_from_file_location('streamlit_app', app_path)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
print('loaded app', hasattr(app, 'make_batch_predictions'))
df = pd.read_csv(root / 'European_Bank.csv')
df.columns = df.columns.str.strip()
print('batch shape', df.shape)
print('batch cols', list(df.columns))
res = app.make_batch_predictions(df.head(5))
print(res[['Predicted probability', 'Predicted Exited']].to_string(index=False))
