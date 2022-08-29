"""This file contains code used in ES20r Sports of Physics,
by Jason Martinez (jmartinez@seas.harvard.edu).

Copyright 2022 Jason Martinez
License: MIT License (https://opensource.org/licenses/MIT)
"""

# Import Libraries:
import pandas as pd                 # Python library for data manipulation and analysis
import matplotlib.pyplot as plt     # Python library for data visualization
import numpy as np                  # Numerical Python   


def prep_data(file_path):
    """
    Description:
        Function for data prep of any Komotion .csv datafile

    Input:
        file_path: string containing the file path location (or URL link) of your .csv

    Output: (returns dataframes from each configuration, in alphabetical order)
        data_a: dataframe with acceleration (a) data
        data_g: dataframe with gyroscipic (g) data - angular velocity
        data_l: dataframe with linear acceleration (l) data
        data_m: dataframe with magnetometer (m) data
        data_r: dataframe with rotation vector (r) data
    """ 
  
    # ---------------- Load Data File-------------------------
    # Load csv content into a Pandas dataframe:
    data = pd.read_csv(file_path, header=None)  # First argument is path to file; `header=None` tells Panda ... 
                                                          # ... that the .csv file does not have a header row with column names

    # Pandas may read column 4 as a column of strings, rather than a float (i.e, decimal number), if it contains missing entries. 
    if data[4].dtypes == 'object':  # if the datatype of column 4 of data is an 'object' (i.e., a string), convert to numerical
        data[4] = pd.to_numeric(data[4], errors='coerce')


    #------------------Separate Data based on configurations---------------
    # Filter 'data' based on contents of column 0 (i.e., a, m, l, r, g)
    data_a = data[data[0] == 'a'].copy()  # Filter 'data' by 'data[0] == a', and copy to a new dataframe 'data_a`.
    data_m = data[data[0] == 'm'].copy()
    data_l = data[data[0] == 'l'].copy()
    data_g = data[data[0] == 'g'].copy()
    data_r = data[data[0] == 'r'].copy()

    # Remove column #0 (no longer required), and column #4 (only for data_a, data_m, data_l, and data_g):
    data_a.drop(columns=[0, 4], inplace=True)   # Remove column '0' and '4'.
    data_m.drop(columns=[0, 4], inplace=True)   # Remove column '0' and '4'.
    data_l.drop(columns=[0, 4], inplace=True)   # Remove column '0' and '4'.
    data_g.drop(columns=[0, 4], inplace=True)   # Remove column '0' and '4'.
    data_r.drop(columns=[0], inplace=True)      # Only remove column '0' for data_r.

    # Rename columns of each dataframe: (Note that order is important and matches what is described in sensor usermanual) 
    data_a.columns = ['ax', 'ay', 'az', 'time'] 
    data_m.columns = ['mx', 'my', 'mz', 'time']
    data_l.columns = ['lx', 'ly', 'lz', 'time']
    data_g.columns = ['gx', 'gy', 'gz', 'time']
    data_r.columns = ['a', 'b', 'c', 'd', 'time'] # a, b, c, d, from quanterion: q = a + bi + cj + dk

    # Reset Index: Make the row numbering (i.e., index) start at 0, 1, 2, ...
    data_a.reset_index(inplace=True, drop=True)
    data_m.reset_index(inplace=True, drop=True)
    data_l.reset_index(inplace=True, drop=True)
    data_g.reset_index(inplace=True, drop=True)
    data_r.reset_index(inplace=True, drop=True)


    # ---------------------Convert quaternion to Euler angles----------------------
    # Compute Roll:
    t0 = 2.0 * (data_r.a * data_r.b + data_r.c * data_r.d)
    t1 = +1.0 - 2.0 * (data_r.b * data_r.b + data_r.c * data_r.c)
    data_r['roll_x'] = np.degrees(np.arctan2(t0, t1))  # Store roll in 'data_r', under a new column named 'roll_x'

    # Compute Pitch:
    t2 = +2.0 * (data_r.a * data_r.c - data_r.d * data_r.b)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    data_r['pitch_y'] = np.degrees(np.arcsin(t2))      # Store pitch in 'data_r', under a new column named 'pitch_y'

    # Compute Yaw:
    t3 = +2.0 * (data_r.a * data_r.d + data_r.b * data_r.c)
    t4 = +1.0 - 2.0 * (data_r.c * data_r.c + data_r.d * data_r.d)
    data_r['yaw_z'] = np.degrees(np.arctan2(t3, t4))    # Store yaw in 'data_r', under a new column named 'yaw_z'

    # ---------------------Return Dataframes----------------------
    return data_a, data_g, data_l, data_m, data_r


def plot_data(file_path, dark_mode=False, quaternion=False):
    """
    Input:
        file_path: string containing the file path location (or URL link) of your .csv
        dark_mode: Boolean variable for dark mode plotting: 'True' for dark mode plotting; 'False' for white background
        quaternion: Boolean variable for plotting Quaternions: 'True' for plotting quaternions; 'False' for plotting Euler angles
    Output: None
        The function renders the figure of your data and saves the image as a .png file. 
    """ 

    # Call prep_data() to prep and tidy the data:
    data_a, data_g, data_l, data_m, data_r = prep_data(file_path)

    
    #----------------------------- Plot Data---------------------------
    # Reset default matplotlib settings:
    plt.rcdefaults()

    # set global parameters for plotting (optional):
    fs = 14        # desired font size for figure text. 
    plt.rcParams["font.family"] = "serif"  # Set font style globally to serif (much nicer than default font style).

    # Set dark mode style on if dark_mode == True:
    if dark_mode:  # True if dark_mode == True
      plt.style.use('dark_background')  # set 'dark_background' sylte on
      edge_color = 'white'  # set color for legend box to white
    else:
      edge_color = 'black'  # set color for legend box to black

    # Find out how many unique configurations exists in the datafile:
    data = pd.read_csv(file_path, header=None)
    if data[4].dtypes == 'object':  # if the datatype of column 4 of data is an 'object' (i.e., a string), convert to numerical
        data[4] = pd.to_numeric(data[4], errors='coerce')
    config = data.loc[:,0].unique() 

    # Determine the number of plots:
    num_plots = len(config) + 1

    # Determine number of rows and columns of plots, based on the sensor configuration:
    if (num_plots == 1):
        num_rows = 1
        num_cols = 1
        fig_size = (9,3) # Figure size in inches (default units)
    elif (num_plots == 2):
        num_rows = 1
        num_cols = 2
        fig_size=(14,3) # Figure size in inches (default units)
    elif (num_plots == 4 or num_plots == 3):
        num_rows = 2
        num_cols = 2
        fig_size=(13,7) # Figure size in inches (default units)
    elif (num_plots == 5 or num_plots == 6):
        num_rows = 3
        num_cols = 2
        fig_size=(13,11) # Figure size in inches (default units)

    # Define figure properties: number of rows fo plots, columns, and overall figure size.
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=fig_size)

    # Initialize counter for row and column:
    cur_row = 0
    cur_col = 0

    if('a' in config):
        ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else, to determine correct definition of 'axes'.
        ax.plot(data_a.time, data_a.ax,'r', label="$a_x$")   # Plots time vs. ax
        ax.plot(data_a.time, data_a.ay, 'y', label="$a_y$")  # Plots time vs. ay
        ax.plot(data_a.time, data_a.az, 'b', label="$a_z$")  # Plots time vs. az
        ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')  # Creates legend.
        ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
        ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
        ax.set_xlabel('Time [s]', fontsize=fs)  # Set x-label
        ax.set_ylabel("Acceleration $[\mathrm{m/s^2}]$", fontsize=fs)  # Set y-label
        ax.minorticks_on()  # Turns minor thicks on (optional)
        ax.grid()           # Shows grid

        cur_col += 1  # Update counter for column:
        if(cur_col > num_cols - 1):  # Update counter for row:
            cur_col = 0
            cur_row +=1

    if('l' in config):
        ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else 
        ax.plot(data_l.time, data_l.lx,'r', label="$l_x$")
        ax.plot(data_l.time, data_l.ly, 'y', label="$l_y$")
        ax.plot(data_l.time, data_l.lz, 'b', label="$l_z$")
        ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
        ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
        ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
        ax.set_xlabel('Time [s]', fontsize=fs)
        ax.set_ylabel("Acceleration $[\mathrm{m/s^2}]$", fontsize=fs)
        ax.minorticks_on()  # Turns minor thicks on (optional)
        ax.grid()           # Shows grid

        cur_col += 1  # Update counter for column:
        if(cur_col > num_cols - 1):
            cur_col = 0
            cur_row +=1

    if('g' in config):
        ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else
        ax.plot(data_g.time, data_g.gx,'r', label="$\omega_x$")
        ax.plot(data_g.time, data_g.gy, 'y', label="$\omega_y$")
        ax.plot(data_g.time, data_g.gz, 'b', label="$\omega_z$")
        ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
        ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
        ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
        ax.set_xlabel('Time [s]', fontsize=fs)
        ax.set_ylabel("Angular Velocity $[\mathrm{rad/s}]$", fontsize=fs)
        ax.minorticks_on()  # Turns minor thicks on (optional)
        ax.grid()           # Shows grid

        cur_col += 1  # Update counter for column:
        if(cur_col > num_cols - 1):
            cur_col = 0
            cur_row +=1

    if('m' in config):
        ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else
        ax.plot(data_m.time, data_m.mx,'r', label="$m_x$")
        ax.plot(data_m.time, data_m.my, 'y', label="$m_y$")
        ax.plot(data_m.time, data_m.mz, 'b', label="$m_z$")
        ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
        ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
        ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
        ax.set_xlabel('Time [s]', fontsize=fs)
        ax.set_ylabel("Magnetic Field  $[\mu \mathrm{T}]$", fontsize=fs)
        ax.minorticks_on()  # Turns minor thicks on (optional)
        ax.grid()           # Shows grid

        cur_col += 1   # Update counter for column:
        if(cur_col > num_cols - 1):
            cur_col = 0
            cur_row +=1

    if('r' in config):
        if quaternion == True:
            ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else
            ax.plot(data_r.time, data_r['a'],'r', label="a")
            ax.plot(data_r.time, data_r['b'], 'y', label="b")
            ax.plot(data_r.time, data_r['c'], 'b', label="c")
            ax.plot(data_r.time, data_r['d'], 'g', label="d")
            ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
            ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax.set_xlabel('Time [s]', fontsize=fs)
            ax.set_ylabel("Value  [-]", fontsize=fs)
            ax.set_ylim([-1, 1])  # Set Y limits
            ax.minorticks_on()    # Turns minor thicks on (optional)
            ax.grid()             # Shows grid
        else:
            ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col] # Ternary logical for if-else
            ax.plot(data_r.time, data_r['roll_x'],'r', label="Roll")
            ax.plot(data_r.time, data_r['pitch_y'], 'y', label="Pitch")
            ax.plot(data_r.time, data_r['yaw_z'], 'b', label="Yaw")
            ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
            ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax.set_xlabel('Time [s]', fontsize=fs)
            ax.set_ylabel("Euler Angles  [degrees $^{\circ}$]", fontsize=fs)
            ax.minorticks_on()  # Turns minor thicks on (optional)
            ax.grid()           # Shows grid

        cur_col += 1  # Update counter for column:
        if(cur_col > num_cols - 1):
            cur_col = 0
            cur_row +=1

    # Plot sample linearity:
    ax = axes[cur_row][cur_col] if (num_plots > 2) else axes[cur_col]
    if('a' in config):
        ax.plot(data_a.time, data_a.index.to_numpy(), 'r', label="$a$")
    if('g' in config):
        ax.plot(data_g.time, data_g.index.to_numpy(),'y', label="$g$")
    if('m' in config):
        ax.plot(data_m.time, data_m.index.to_numpy(),'b', label="$m$")
    if('r' in config):
        ax.plot(data_r.time, data_r.index.to_numpy(), 'g', label="$r$")
    if('l' in config):
        ax.plot(data_l.time, data_l.index.to_numpy(),'k', label="$l$")

    ax.legend(fontsize=fs, ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')
    ax.tick_params(which='major', labelsize=fs, width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
    ax.tick_params(which='minor', labelsize=fs, width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
    ax.set_xlabel('Time [s]', fontsize=fs)
    ax.set_ylabel("Sample Counts", fontsize=fs)
    from matplotlib.ticker import StrMethodFormatter  # required import to ensure interger values on y-axis of 'Sample Count'
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # Fancy code to ensure integer values on y-axis of 'Sample Count'
    ax.minorticks_on()  # Turns minor thicks on (optional)
    ax.grid()           # Shows grid

    fig.tight_layout(pad=2.0)  # To add padding between subplots.
    plt.show()                 # To show the plot.
    fig.savefig(file_path.split('/')[-1][:-4] + '_Figure.png', dpi=400, bbox_inches='tight')   # To save the entire figure.