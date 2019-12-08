import numpy as np
import pandas as pd
import math

x0 = np.random.rand(1000) * 180
x1 = np.random.rand(1000) * 180
s = [math.sin(math.radians(i+s)) for i, s in zip(x0, x1)]
c = [math.cos(math.radians(i+s)) for i, s in zip(x0, x1)]

df = pd.DataFrame({'x0':x0, 'x1':x1, 'sin':s, 'cos':c})
df.to_csv('../input/data.csv')
