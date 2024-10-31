### HW5 Geostatistics ###
# Created by: Anna Grunes
# 10/8/24

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools as it
import matplotlib.pyplot as plt 

#from geostats_functions import * #pulling in functions from geostats_functions.py file
## To see functions, open geostats_functions.py

#Information regarding these functions can be found in the docstrings (type 'help(my_function)')



#########---------------------------------------#####################
###### USER INPUT ########
#Adjust this for number of bins specified
n=30
###### END USER INPUT #######
#########---------------------------------------#####################




##### Program starts here ############
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    '''Creating scatter plots with histograms on both the 
    x and y axes'''
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha = 0.2,s=10, lw=0)

    ax_histx.hist(x, bins=60)
    ax_histy.hist(y, bins=60, orientation='horizontal')


pairs_df=pd.read_csv('pairs_df.csv')

#Plotting Distance vs. Semivariance
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x=pairs_df['Distance'], y=pairs_df['Semivariance'], ax=ax, ax_histx=ax_histx, ax_histy=ax_histy)
fig.suptitle(f'Distance vs. Semivariance\nYou specifed {n} bins. Click {n} times for user-specified bins')

print(f'after {n} clicks')
bin_breaks = plt.ginput(n=n, timeout = 90)
print(bin_breaks)
#print(type(bin_creaks))
#plt.tight_layout()
with open('bin_breaks_interactive.txt', 'w+') as f:
    
    # write elements of list
    for item in bin_breaks:
        # Convert the tuple to a string and write it to the file
        f.write(f"{item}\n")  # Each tuple on a new line
    
    print("File written successfully")


# close the file
f.close()    
plt.show()

