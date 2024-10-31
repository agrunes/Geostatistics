import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

#Pearson's Correlation Coefficient: 
def pearson_corr(df,x,y):
    ''' Calculate Pearson's correlation coefficient for a given dataset in 
            dataframe 'df'
        df = dataframe with values 
        x = name of independent variable column
        y = name of dependent variable column
        '''
    #len_check = len(df[x])==len(df[y])
    #print(f'Lengths are equal = {len_check}')
    n = len(df[x])
    sum_lst = 0
    m_x = np.mean(df[x]) #mean of x
    m_y = np.mean(df[y]) #Mean of y
    stdev_x = np.std(df[x]) #Stdev of x
    stdev_y = np.std(df[y]) #stdev of y

    for i,row in df.iterrows(): #iterate for each i in the df
        term1 = row[x]-m_x
        term2 = row[y]-m_y
        full = (term1*term2)/(stdev_x*stdev_y)
        sum_lst=sum_lst+full #summation
    rho = sum_lst/n 
    return rho

def covariance(df,x,y):
    ''' Calculate covariance for a given dataset in 
            dataframe 'df'
        df = dataframe with values 
        x = name of independent variable column
        y = name of dependent variable column
        '''
    n = len(df[x])
    sum_lst = 0
    m_x = np.mean(df[x]) #mean of x
    m_y = np.mean(df[y]) #mean of y
    for i,row in df.iterrows():#iterate for each i in the df
        term1 = row[x]-m_x
        term2 = row[y]-m_y
        full = (term1*term2)
        sum_lst=sum_lst+full #Summation
    cv = sum_lst/n
    return cv

def calc_semivariance(df,x,y):
    ''' Calculate semivariance for a given dataset in 
            dataframe 'df'
        df = dataframe with values 
        x = name of independent variable column
        y = name of dependent variable column
        '''
    n = len(df[x]) #Number of pairs in the lag dataset
    sum_lst = 0
    for i,row in df.iterrows():
        term1 = (row[y]-row[x])**2
        sum_lst=sum_lst+term1
    semivar = sum_lst/(2*n)
    return semivar


def lag_dist(df,x,y,h,theta,primary, cross, secondary):
    ''' Output the 'near' and 'far' vectors (as a dataframe df_lag) 
            for a given spatial dataset 
            in dataframe 'df'.
    
    df = dataframe with values 
    x = Column name of x-coord
    y = Column name of y-coord
    h = lag distance
    theta = angle of interest *calculates for either 0 or 90 degrees
    primary = parameter of interest (i.e. Permeability, Water Resistivity)
    cross = True or False (Is this a cross-statistic or not?)
    secondary = secondary parameter for cross-statistic
    '''

    dir_dict={0:x,90:y}
    direction_dict = {0:y, 90:x}
    dir_2 = dir_dict[theta]
    direction = direction_dict[theta]
    near = []
    near_prim = []
    near_sec=[]
    far = []
    far_prim=[]
    far_sec = []
    for y_val in df[direction].unique(): #specify direction of lag distance calculations
        x_sub = df[df[direction]==y_val] #gather subset for each unique value

        for i,row in x_sub.iterrows():
            x_val = row[dir_2]
            if x_val+h in list(x_sub[dir_2]):
                near.append(row[primary])
                x_h_sub = x_sub[x_sub[dir_2]==x_val+h]
                if cross==False:
                    far.append(float(x_h_sub[primary]))
                elif cross==True:
                    far_sec.append(float(x_h_sub[secondary]))
                    near_prim.append(row[primary])
                    far_prim.append(float(x_h_sub[primary]))
                    near_sec.append(row[secondary])

    #Create DF so this is compatible with other functions 
    if cross==False: 
        df_lag = pd.DataFrame({'Near':near, 'Far':far})
    elif cross==True:
        df_lag = pd.DataFrame({'Near (Primary)':near_prim, 
                                'Near (Secondary)':near_sec,
                                'Far (Primary)':far_prim,
                                'Far (Secondary)':far_sec})                               

    return df_lag

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    '''Creating scatter plots with histograms on both the 
    x and y axes'''
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha = 0.5,s=10)

    ax_histx.hist(x, bins=10)
    ax_histy.hist(y, bins=10, orientation='horizontal')


#### HW4 Additions ####
def calc_cross_semivariance(df,x_near,x_far, y_near, y_far):
    ''' Calculate cross semivariance for a given dataset in 
            dataframe 'df'
        df = dataframe with values 
        x_near = name of PRIMARY variable NEAR column
        x_far = name of PRIMARY variable LAGGED column
        y_near = name of SECONDARY variable NEAR column
        y_far = name of SECONDARY variable LAGGED column

        '''
    n = len(df[x_near]) #Number of pairs in the lag dataset
    sum_lst = 0
    # x_far_mean = np.mean(df[x_far])
    # y_far_mean = np.mean(df[y_far])

    for i,row in df.iterrows():
        term1 = row[x_near]-row[x_far] #Near - far (Primary Variable)
        term2 = row[y_near]-row[y_far] #Near - far (Secondary Variable)
        mult = term1*term2 #Multiply together
        sum_lst=sum_lst+mult #Sigma
    semivar = sum_lst/(2*n) #Divide by 2*n(h)
    return semivar

    
def calc_euclidean_dist(x1,y1,x2,y2,**kwargs):
    ''' Function to calculate Euclidean Distance between two points
    x1,y1 = x,y coordinate for pt1
    x2,y2 = x,y coordinates for pt2
    **Optional: 
    z1,z2 = z-coordinates for pt1 and pt2
    '''

    try:
        z1=kwargs['z1']
        z2=kwargs['z2']
        dist = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        #print('Calculating with z-coordinate')
    except:
        dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
        #print('Calculating with x,y coordinates')

        #print('z-coordinate not given or not in correct format.\n input as z1=val,z2=val')

    return dist

def calc_semivar_individual(near,far):
    '''Calculating Semivariance for a given pair of points'''

    semivar = ((near-far)**2)/2
    return semivar

def apply_spherical_model(omega,a, df):
    h = np.linspace(0,df['Distance'].max(), 1000)
    model_continuous = []
    k=3/2
    for val in h:
        if val<a:
            model = omega*((k*(val/a))-(0.5*(val/a)**3))
        else:
            model = omega
        model_continuous.append(model)
    df_model = pd.DataFrame({'Distance':h,'model':model_continuous})
    name = 'Spherical Model'
    return df_model,name

def apply_exponential_model(omega,a, df):
    h = np.linspace(0,df['Distance'].max(), 1000)
    model_continuous = []
    model=omega*(1-np.exp(-h/a))
    df_model = pd.DataFrame({'Distance':h,'model':model})
    name = 'Exponential Model'
    return df_model,name

def apply_gaussian_model(omega,a, df):
    h = np.linspace(0,df['Distance'].max(), 1000)
    model_continuous = []
    model=omega*(1-np.exp(-(h/a)**2))
    df_model = pd.DataFrame({'Distance':h,'model':model})
    name = 'Gaussian Model'
    return df_model,name

def semivariogram_equal_points(pairs_df, n_bins, fit_model, model_func, **model_pars):
    ''''Semivariogram with equal number of points in each bin
    pairs_df = DataFrame with pairs,distance, and corresponding semivariance
    n_bins = Number of equal frequency bins
    fit_model = Fit a model to the semivariogram (True or False)
    model_func = Which model function to fit
    if fit_model == True, add in arguments for range (a) and sill (omega)
    '''
    pairs_df_sorted = pairs_df.sort_values(by='Distance', ignore_index=True) #Sorting pairs by distance
    #Use qcut to put equal number of pairs in bins
    #n_bins = number of equal frequency groups
    #pairs_df_sorted['Bin Int.']=pd.qcut(x=pairs_df_sorted['Distance'],q=n_bins)
    pairs_df_sorted['Bin Int.']=pd.qcut(x=pairs_df_sorted['Distance'].rank(method='first'),q=n_bins)
    bin_counts = pairs_df_sorted['Bin Int.'].value_counts() #Seeing how many values in each bin

    bin_df=pd.DataFrame()
    j=0
    for val in pairs_df_sorted['Bin Int.'].unique(): #For each bin, calculate the statistics
        j+=1
        a = pairs_df_sorted[pairs_df_sorted['Bin Int.']==val]
        mean_dist = a['Distance'].mean()
        mean_semivariance=a['Semivariance'].mean()
        bin_n = len(a)
        bin_df.loc[val,'mn_dist']=mean_dist
        bin_df.loc[val,'bin_n']=bin_n
        bin_df.loc[val,'mn_semivar']=mean_semivariance
        bin_df.loc[val,'Bin #']=j


    if fit_model == False: #Plotting without model
        #Plotting experimental Semivariogram
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.9,lw=0)
        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        ax.set_ylim(280,290)
        ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5)
        ax.axhline(bin_df['bin_n'].mean(), label = f'Avg n={bin_df['bin_n'].mean():.2f}', c='Red')
        ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            ax.axvline(interval.left, alpha = 0.2)
        ax.set_xlabel('Distance [mm]')
        ax.set_title('Bin Boundaries')


        plt.suptitle('Experimental Semivariogram [equal # of points in each bin] \n Unequal Bin Sizes')
        plt.tight_layout()
        plt.show()
    if fit_model == True: #Plotting with model fitted
        omega = model_pars['omega']
        a = model_pars['a']
        #FItting model depending on the model input function given
        model_spher,name = model_func(omega=omega, a=a, df=pairs_df) #Generating model from given range and sill

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.9,lw=0,label = 'Semivariogram')
        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        ax.plot(model_spher['Distance'],model_spher['model'], label=f'{name}', lw=5, color='Red',alpha = 0.3)
        ax.axvline(a, ls='--',color='#fdae6b', alpha = 0.8)
        ax.axhline(omega,ls='--',color='#8c6bb1', alpha = 0.8 )
        ax.text(a+8,omega+5000,f'Range = {a} mm \nSill = {omega}')
        ax.legend()
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        ax.set_ylim(280,290)
        ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5)
        ax.axhline(bin_df['bin_n'].mean(), label = f'Avg n={bin_df['bin_n'].mean():.2f}', c='Red')
        ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            try:
                ax.axvline(interval.left, alpha = 0.2)
            except:
                pass
        ax.set_xlabel('Distance [mm]')
        ax.set_title('Bin Boundaries')


        plt.suptitle(f'Experimental Semivariogram [equal # of points in each bin] \n model = {name}')
        plt.tight_layout()
        plt.show()

#Function to calculate bin_df with different         
def bin_width_calcs(bin_width, pairs_df_sorted):
    #print(f'Bin widths set to {user_bin} (fixed equal width)')
    #Use pd.cut to put into bins of equal widths
    pairs_df_sorted['Bin Int.']=pd.cut(x=pairs_df_sorted['Distance'],bins=bin_width, duplicates = 'drop')
    bin_counts = pairs_df_sorted['Bin Int.'].value_counts() #Seeing how many values in each bin
    bin_df=pd.DataFrame()
    for val in pairs_df_sorted['Bin Int.'].unique():
        a = pairs_df_sorted[pairs_df_sorted['Bin Int.']==val]
        mean_dist = a['Distance'].mean()
        mean_semivariance=a['Semivariance'].mean()
        bin_n = len(a)
        bin_df.loc[val,'mn_dist']=mean_dist
        bin_df.loc[val,'bin_n']=bin_n
        bin_df.loc[val,'mn_semivar']=mean_semivariance

    return bin_df

def semivariogram_equal_bin_widths(pairs_df, bin_width, fit_model, model_func, **model_pars):
    ''''Semivariogram with equal width of bins
    pairs_df = DataFrame with pairs,distance, and corresponding semivariance
    bin_width = User-specified width of bins (singel value or a list)
    fit_model = Fit a model to the semivariogram (True or False)
    model_func = Which model function to fit
    if fit_model == True, add in arguments for range (a) and sill (omega)
    '''
    pairs_df_sorted = pairs_df.sort_values(by='Distance', ignore_index=True) #Sorting pairs by distance
    #bin_df = bin_width_calcs(bin_width,pairs_df_sorted)


    if fit_model == False:
        #Plotting experimental Semivariogram
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        try:
            for user_bin in bin_width: #looping through various bin sizes
                bin_df=bin_width_calcs(user_bin,pairs_df_sorted)
                ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.5,lw=0, label = f'Bin width = {user_bin}')
        except:
            bin_df=bin_width_calcs(bin_width,pairs_df_sorted)
            ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.5,lw=0, label = f'Bin width = {bin_width}')

        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.legend()
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        try:
            for user_bin in bin_width: #looping through various bin sizes
                bin_df=bin_width_calcs(user_bin,pairs_df_sorted)
                ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5, label = f'bin width = {user_bin}')
        except:
            bin_df=bin_width_calcs(bin_width,pairs_df_sorted)
            ax.plot(bin_df['mn_dist'],bin_df['bin_n'],zorder=2,alpha=0.5, label = f'Bin width = {bin_width}')

        ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            ax.axvline(interval.left, alpha = 0.2)
        ax.set_xlabel('Distance [mm]')
        try:
            ax.set_title(f'Bin Boundaries (bin width = {bin_width[0]})')
        except:
            ax.set_title(f'Bin Boundaries (bin width = {bin_width})')

        plt.suptitle(f'Experimental Semivariogram [equal bin sizes of user specification] \n Unequal n-values in bin')
        plt.tight_layout()
        plt.show()
    if fit_model == True:

        omega = model_pars['omega']
        a = model_pars['a']
        #FItting model depending on the model input function given
        model_spher,name = model_func(omega=omega, a=a, df=pairs_df) #Generating model from given range and sill

        #Plotting experimental Semivariogram
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        try:
            for user_bin in bin_width: #looping through various bin sizes
                bin_df=bin_width_calcs(user_bin,pairs_df_sorted)
                ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.5,lw=0, label = f'Bin width = {user_bin}')
        except:
            bin_df=bin_width_calcs(bin_width,pairs_df_sorted)
            ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.5,lw=0, label = f'Bin width = {bin_width}')
        ax.plot(model_spher['Distance'],model_spher['model'], label=f'{name}', lw=5, color='Red',alpha = 0.3)

        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        # ax.set_xlabel('Distance [mm]')
        ax.axvline(a, ls='--',color='#fdae6b', alpha = 0.8)
        ax.axhline(omega,ls='--',color='#8c6bb1', alpha = 0.8 )
        ax.text(a+8,omega+5000,f'Range = {a} mm \nSill = {omega}')
        ax.tick_params(axis='x',labelbottom=False)
        ax.legend()
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        try: #Makes this work with either a list of values or a single value
            for user_bin in bin_width: #looping through various bin sizes
                bin_df=bin_width_calcs(user_bin,pairs_df_sorted)
                ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5, label = f'bin width = {user_bin}')
        except:
            bin_df=bin_width_calcs(bin_width,pairs_df_sorted)
            ax.plot(bin_df['mn_dist'],bin_df['bin_n'],zorder=2,alpha=0.5, label = f'Bin width = {bin_width}')

        ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            ax.axvline(interval.left, alpha = 0.2)
        ax.set_xlabel('Distance [mm]')
        try:
            ax.set_title(f'Bin Boundaries (bin width = {bin_width[0]})')
        except:
            ax.set_title(f'Bin Boundaries (bin width = {bin_width})')

        plt.suptitle(f'Experimental Semivariogram [equal bin sizes of user specification] \n model = {name}')
        plt.tight_layout()
        plt.show()

def semivariogram_interactive_bins(pairs_df, bin_breaks_file, fit_model, model_func, **model_pars):
    ''''Semivariogram with bins defined by an interactive file, which outputs bin_breaks_file (a list of coordinates chosen by user)
    pairs_df = DataFrame with pairs,distance, and corresponding semivariance
    bin_breaks_file = a .txt file with a list of coordinates chosen (new line separated) [ex. (17.277840594722264, 12106.56807000002)]
    fit_model = Fit a model to the semivariogram (True or False)
    model_func = Which model function to fit
    if fit_model == True, add in arguments for range (a) and sill (omega)
    '''

    # Initialize an empty list to store tuples
    bin_breaks_lst = []

    # Open the file 
    with open(bin_breaks_file, 'r') as file:
        for line in file:

            # Evaluate the string back to a tuple and append it to the lists
            bin_breaks_lst.append(eval(line))

    first = bin_breaks_lst[0][0]
    new_tuple=(0,first)
    bin_breaks_lst.insert(0, new_tuple)
    bin_breaks = [x for x, y in bin_breaks_lst]
    bin_breaks.sort() #it doesnt matter in what order you click!

    pairs_df_sorted = pairs_df.sort_values(by='Distance', ignore_index=True) #Sorting pairs by distance
    # Using cut to divide the dataset into bins based on user interaction
    pairs_df_sorted['Bin Int.']=pd.cut(x=pairs_df_sorted['Distance'],bins=bin_breaks, duplicates = 'drop')
    bin_counts = pairs_df_sorted['Bin Int.'].value_counts() #Seeing how many values in each bin
    bin_df=pd.DataFrame()
    j=0
    for val in pairs_df_sorted['Bin Int.'].unique():
        j+=1
        a = pairs_df_sorted[pairs_df_sorted['Bin Int.']==val]
        mean_dist = a['Distance'].mean()
        mean_semivariance=a['Semivariance'].mean()
        bin_n = len(a)
        bin_df.loc[val,'mn_dist']=mean_dist
        bin_df.loc[val,'bin_n']=bin_n
        bin_df.loc[val,'mn_semivar']=mean_semivariance
        bin_df.loc[val,'Bin #']=j

    #Plotting experimental Semivariogram
    if fit_model == False:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.9,lw=0)
        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5)
        #ax.axhline(bin_df['bin_n'].mean(), label = f'Avg n={bin_df['bin_n'].mean():.2f}', c='Red')
        #ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            try:
                ax.axvline(interval.left, alpha = 0.2)
            except:
                pass
        ax.set_xlabel('Distance [mm]')
        ax.set_title('Bin Boundaries')


        plt.suptitle(f'Experimental Semivariogram\nUser Specified Bins n_bins={len(bin_breaks)-1}')
        plt.tight_layout()
        plt.show()                                          

    if fit_model == True:
        omega = model_pars['omega']
        a = model_pars['a']
        #FItting model depending on the model input function given
        model_spher,name = model_func(omega=omega, a=a, df=pairs_df) #Generating model from given range and sill

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8),gridspec_kw={'height_ratios': [4, 1, 1]})
        axes = axes.flatten()
        ax=axes[0]
        ax.grid(alpha=0.4)
        ax.set_ylim(-1000,30000)
        ax.scatter(bin_df['mn_dist'],bin_df['mn_semivar'],zorder=2,alpha=0.9,lw=0,label = 'Semivariogram')
        ax.scatter(pairs_df_sorted['Distance'],pairs_df_sorted['Semivariance'], alpha = 0.1, color='#bdbdbd',lw=0,zorder=1)
        ax.plot(model_spher['Distance'],model_spher['model'], label=f'{name}', lw=5, color='Red',alpha = 0.3)
        ax.axvline(a, ls='--',color='#fdae6b', alpha = 0.8)
        ax.axhline(omega,ls='--',color='#8c6bb1', alpha = 0.8 )
        ax.text(a+8,omega+5000,f'Range = {a} mm \nSill = {omega}')
        ax.legend()
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Semivariance')
        ax=axes[1]
        ax.grid(alpha=0.4)
        ax.plot(bin_df['mn_dist'],bin_df['bin_n'], alpha=0.5)
        #ax.axhline(bin_df['bin_n'].mean(), label = f'Avg n={bin_df['bin_n'].mean():.2f}', c='Red')
        ax.legend()
        # ax.set_xlabel('Distance [mm]')
        ax.tick_params(axis='x',labelbottom=False)
        ax.set_ylabel('Number of Points')
        ax.set_title('Number of Data Points in Each Bin')
        ax=axes[2]
        for interval in bin_df.index:
            try:
                ax.axvline(interval.left, alpha = 0.2)
            except:
                pass
        ax.set_xlabel('Distance [mm]')
        ax.set_title('Bin Boundaries')


        plt.suptitle(f'Experimental Semivariogram\nUser Specified Bins n_bins={len(bin_breaks)-1}')
        plt.tight_layout()
        plt.show()
