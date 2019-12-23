import pandas as pd
import numpy as np

x=np.array([1,2,3])
y=np.array([4,5,6])

data1 = pd.DataFrame(x)
data1.to_csv('test.csv', mode='a', header=False, index=False)
data2 = pd.DataFrame(y)
data2.to_csv('test.csv', mode='a', header=False, index=False)
