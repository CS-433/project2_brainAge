import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('_mpl-gallery')

# make data:
data = pd.read_excel('data.xlsx', 'Timepoint 1')

print(data)
# plot
#fig, ax = plt.subplots()

#ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       #ylim=(0, 8), yticks=np.arange(1, 8))

#plt.show()