import matplotlib.cm as cm
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_results(title, data, predictions):
  colors = {-1 : 'g',
                   0 : 'c',
                   1 : 'b',
                   2 : 'r'

                   }
  plt.title(title)
  data = np.array(data)
  temp, sal = data[:,0], data[:,1]

  label_color = [colors[l] for l in predictions]
  plt.scatter(temp, sal, c=label_color)
  truelist = []
  for val in colors:
      truelist.append(mpatches.Patch(color=colors[val], label=val))
  plt.legend(handles=truelist)
  plt.show()

def plot_data_points(valid_points, anomalies):
    plt.scatter(valid_points[:,0], valid_points[:,1])
    plt.scatter(anomalies[:,0], anomalies[:,1])
    plt.xlabel("Temperature (degrees C)")
    plt.ylabel("Salinity (PPT)")
    plt.show()
