import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('result_folder/pred_label_1024.txt', header=0, sep=' ')

confusion = confusion_matrix(df.real, df.predict)
print(confusion)
