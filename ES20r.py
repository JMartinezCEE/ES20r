"""
This file contains code used in ES20r Sports of Physics,
by Jason Martinez (jmartinez@seas.harvard.edu).
https://www.seas.harvard.edu/computing-engineering-education

Last Modified: 09/23/2022
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
    # Import Libraries:
    import pandas as pd                 # Python library for data manipulation and analysis
    import matplotlib.pyplot as plt     # Python library for data visualization
    import numpy as np                  # Numerical Python

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


def rot_mat(a, b, c, d):
    """
    Input:
        a: real value component of a quaternion q = a + bi + cj + dk
        b: 1st imaginary value component 'b', of a quaternion q = a + bi + cj + dk 
        c: 2nd imaginary value component 'c', of a quaternion q = a + bi + cj + dk 
        d: 3rd imaginary value component 'd', of a quaternion q = a + bi + cj + dk 
    Output:
        rotMat: 3x3 NumPy array with rotation matrix R.
    """

    # Import Modules:
    import numpy as np

    # Define components of rotation matrix R:
    c00 = a ** 2 + b ** 2 - c ** 2 - d ** 2
    c01 = 2 * (b * c - a * d)
    c02 = 2 * (b * d + a * c)
    c10 = 2 * (b * c + a * d)
    c11 = a ** 2 - b ** 2 + c ** 2 - d ** 2
    c12 = 2 * (c * d - a * b)
    c20 = 2 * (b * d - a * c)
    c21 = 2 * (c * d + a * b)
    c22 = a ** 2 - b ** 2 - c ** 2 + d ** 2

    rotMat = np.array([[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]])
    return rotMat

def rot_mat_Euler(yaw_z=0, pitch_y=0, roll_x=0, intrinsic=True, sequence='zyx', radians=False):
    """
    Input:
        yaw_z: Float value containing the yaw in [deg] or [rad]; rotation about z-axis  
        pitch_y: Float value containing the pitch in [deg] or [rad]; rotation about y-axis    
        roll_x:  Float value containing the roll in [deg] or [rad]; rotation about x-axis
        intrinsic: Boolean value for intrinsic angles: 'True' for intrinsic angles, 'False' for extrinsic angles 
        sequence: String containing the Euler angle rotation sequence; e.g. 'xyz', or 'XYZ'
        radians: Boolean value for whether radians used for yaw, pitch, and roll: 'True' if using radians, 'False' if using degrees.
    Output:
        rotMat: 3x3 NumPy array with rotation matrix R.
    """

    # Import libraries
    import numpy as np

    # Lowercase the sequence order
    sequence = sequence.lower()   

    # Ensure sequence is unique (i.e., a Tait-Brian Euler angle) and there are three entries
    if (sequence.count('x') != 1) or (sequence.count('y') != 1) or (sequence.count('z') != 1): 
      print('ERROR: The sequence is not correct!')
      return # End Function

    # Convert sequences from Extrinsic to Intrinsic if 'intrinsic==False'
    if intrinsic == False:  # Angles are Extrinsic and must be converted to Intrinsic
        if sequence == 'zyx': sequence = 'xyz'
        if sequence == 'xzy': sequence = 'yzx'
        if sequence == 'yxz': sequence = 'zxy'
        if sequence == 'yzx': sequence = 'xzy'
        if sequence == 'xyz': sequence = 'zyx'
        if sequence == 'zxy': sequence = 'yxz'

    # Convert yaw, pitch, and roll to radians if 'radians==False'
    if radians == False:    # angles input in degrees
        yaw_z = np.radians(yaw_z)
        pitch_y = np.radians(pitch_y)
        roll_x = np.radians(roll_x)

    # Compute cosine and sine angles:
    cx = np.cos(roll_x)
    sx = np.sin(roll_x)
    cy = np.cos(pitch_y)
    sy = np.sin(pitch_y)
    cz = np.cos(yaw_z)
    sz = np.sin(yaw_z)


    # Compute Elemental Rotations matrices:
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]])
    Ry = np.array([[cy, 0, sy],[0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0], [0, 0, 1]])
    
    # Compute final Rotation matrix (sequence dependent)
    if sequence == 'zyx': R = Rz @ Ry @ Rx
    if sequence == 'xzy': R = Rx @ Rz @ Ry  
    if sequence == 'yxz': R = Ry @ Rx @ Rz
    if sequence == 'yzx': R = Ry @ Rz @ Rx
    if sequence == 'xyz': R = Rx @ Ry @ Rz
    if sequence == 'zxy': R = Rz @ Rx @ Ry
    
    return R

def view_quat(a=1, b=0, c=0, d=0, dark_mode=False, swap_xy=False):
    """
    Input:
        a: real value component of a quaternion q = a + bi + cj + dk
        b: 1st imaginary value component 'b', of a quaternion q = a + bi + cj + dk 
        c: 2nd imaginary value component 'c', of a quaternion q = a + bi + cj + dk 
        d: 3rd imaginary value component 'd', of a quaternion q = a + bi + cj + dk 
        dark_mode: Boolean variable for dark mode plotting: 'True' for dark mode plotting; 'False' for white background
        swap_xy: Boolean variable for swapping x-axis and y-axis in 3D view of the sensor
    Output:
        The function renders the figure of your quaternion and saves the image as a .png file.
    """
    # Import Modules:
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.pyplot as plt

    # Normalize quaternion (input must be a unit quaternion)
    mag = np.sqrt(a**2 + b**2 + c**2 + d**2)
    if mag == 0.0:
      print("ERROR: Can't Enter a Zero Quaternion.")
      return # End Function

    a_norm = a/mag
    b_norm = b/mag
    c_norm = c/mag
    d_norm = d/mag


    # Reset default matplotlib settings:
    plt.rcdefaults()

    # set global parameters for plotting (optional):
    plt.rcParams['figure.figsize'] = [14, 6]
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.family"] = "serif"  # Set font style globally to serif (much nicer than default font style).
    plot_limit = 8  # constant for x-axis, y-axis, and z-axis length limits

    # Set dark mode style on if dark_mode == True:
    if dark_mode:  # True if dark_mode == True
      plt.style.use('dark_background')  # set 'dark_background' sylte on
      edge_color = 'white'  # set color for legend box to white
      sensor_color = '#990000' # crimson
      sensor_alpha = 0.6       # alpha transparency between 0 to 1
      sensor_edge_color = 'w'  # white
      x_axis_color = '#ff073a' # neon red
      y_axis_color = '#04d9ff' # neon blue
      z_axis_color = '#39FF14' # neon green
      mag_north_color = 'w'    # white for magnetic north arrow
      text_background_alpha = 0.7
      text_color = 'w'
      text_background = 'k'
    else:         # if dark_mode == False 
      edge_color = 'black'  # set color for legend box to black
      sensor_color = sensor_color = '#990000' # crimson
      sensor_alpha = 0.2
      sensor_edge_color = 'k' # black
      x_axis_color = 'r' # red
      y_axis_color = 'b' # blue
      z_axis_color = 'g' # green
      mag_north_color = 'k'    # white for magnetic north arrow
      text_background_alpha = 0.6
      text_color = 'k'
      text_background = 'w'

    # Define sensor dimensions for plotting:
    width = 4     # y-axis dimension
    height = 1    # z-axis dimension
    depth = 6     # x-axis dimension

    # Define 8 points of rectangular prism:
    pt1 = np.array([-depth/2, -width/2, -height/2 ])
    pt2 = np.array([depth/2, -width/2, -height/2 ])
    pt3 = np.array([depth/2, width/2, -height/2 ])
    pt4 = np.array([-depth/2, width/2, -height/2])
    pt5 = np.array([-depth/2, -width/2, height/2])
    pt6 = np.array([depth/2, -width/2, height/2 ])
    pt7 = np.array([depth/2, width/2, height/2])
    pt8 = np.array([-depth/2, width/2, height/2])

    # define 6 faces of rectangular prism:
    long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
    long_side_face2 = [[pt3, pt7,  pt8, pt4]]
    short_side_face1 = [[pt3, pt2, pt6, pt7]]
    short_side_face2 = [[pt1, pt5, pt8, pt4]]
    bottom_face = [[pt1, pt2, pt3, pt4]]
    top_face = [[pt5, pt6, pt7, pt8]]

    # Define vectors representing coordinate axes x, y, and z:
    arrow_length = depth/2*1.9
    x_axis = np.array([arrow_length, 0, 0])
    y_axis = np.array([0, arrow_length, 0 ])
    z_axis = np.array([ 0, 0, arrow_length ])

    # Define vectors for holding text for each axis:
    x_text = np.array([arrow_length+1, 0, 0])
    y_text = np.array([0, arrow_length+1, 0 ])
    z_text = np.array([ 0, 0, arrow_length+1 ])

    # =============Create Figure 1 of 2: Original sensor orientation================
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot each of the 6 faces of the rectangular prism:
    ax.add_collection3d(Poly3DCollection(long_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(long_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(short_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(short_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(bottom_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(top_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))

    # Draw 3 arrows for each local coordinate axis:
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color=x_axis_color, linewidth=2, linestyle='-') # x-axis
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color=y_axis_color, linewidth=2, linestyle='-') # y-axis
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color=z_axis_color, linewidth=2, linestyle='-') # z-axis

    t = ax.text(x_text[0], x_text[1], x_text[2], s='x', color=x_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax.text(y_text[0], y_text[1], y_text[2], s='y', color=y_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax.text(z_text[0], z_text[1], z_text[2], s='z', color=z_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

     # Magnetic north and gravity
    if swap_xy:
        start_pt = [6, -6, -6]
        text_position = [6, -5.5 + arrow_length, -6.5]
    else:
        start_pt = [6, -6, -6]
        text_position = [5.5, -4.5 + arrow_length, -6.5]

    # Magnetic north arrow:
    ax.quiver(start_pt[0], start_pt[1], start_pt[2],  0 ,arrow_length, 0, color=mag_north_color, linewidth=1.6, linestyle='-') # z-axis

    # Text for 'N' near the magnetic north arrow:
    t = ax.text(text_position[0], text_position[1], text_position[2], s='N', fontsize=14, color=text_color, fontfamily='serif' , fontweight='bold')
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    # Define axis labels and title:
    ax.set_xlabel('Global X-axis')
    ax.set_ylabel('Global Y-axis')
    ax.set_zlabel('Global Z-axis')
    ax.set_title('Original Orientation', fontsize='13')

    # Get rid of number on axis:
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Set plot limits based on 'plot_limit' variable:
    ax.set_xlim3d(-plot_limit, plot_limit)
    ax.set_ylim3d(-plot_limit, plot_limit)
    ax.set_zlim3d(-plot_limit, plot_limit)

    if swap_xy:
        ax.view_init(30, 50) # view

    #=====================Rotation==================
    # Define rotation matrix based on quaternion values a, b, c, and d:
    R = rot_mat(a_norm, b_norm, c_norm, d_norm)

    # ================Rotate all points by rotation matrix R===============
    pt1 = R @ pt1
    pt2 = R @ pt2
    pt3 = R @ pt3
    pt4 = R @ pt4
    pt5 = R @ pt5
    pt6 = R @ pt6
    pt7 = R @ pt7
    pt8 = R @ pt8

    x_axis = R @ x_axis
    y_axis = R @ y_axis
    z_axis = R @ z_axis

    x_text = R @ x_text
    y_text = R @ y_text
    z_text = R @ z_text

    # Redefine faces after rotation:
    long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
    long_side_face2 = [[pt3, pt7,  pt8, pt4]]
    short_side_face1 = [[pt3, pt2, pt6, pt7]]
    short_side_face2 = [[pt1, pt5, pt8, pt4]]
    bottom_face = [[pt1, pt2, pt3, pt4]]
    top_face = [[pt5, pt6, pt7, pt8]]

    # =============Create Figure 2 of 2: Transformed sensor orientation================
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot each of the 6 faces of the rectangular prism:
    ax2.add_collection3d(Poly3DCollection(long_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(long_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(short_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(short_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(bottom_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(top_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))

    # Draw 3 arrows for each local coordinate axis:
    ax2.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color=x_axis_color, linewidth=2, linestyle='-') # x-axis
    ax2.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color=y_axis_color, linewidth=2, linestyle='-') # y-axis
    ax2.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color=z_axis_color, linewidth=2, linestyle='-') # z-axis

    t = ax2.text(x_text[0], x_text[1], x_text[2], s='x', color=x_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax2.text(y_text[0], y_text[1], y_text[2], s='y', color=y_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax2.text(z_text[0], z_text[1], z_text[2], s='z', color=z_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    # Define axis labels and title:
    ax2.set_xlabel('Global X-axis')
    ax2.set_ylabel('Global Y-axis')
    ax2.set_zlabel('Global Z-axis')
    ax2.set_title('Transformed Orientation', fontsize='13')

    # Get rid of number on grids
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.axes.zaxis.set_ticklabels([])

    # Get rid of colored axes planes
    # First remove fill
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax2.xaxis.pane.set_edgecolor('w')
    ax2.yaxis.pane.set_edgecolor('w')
    ax2.zaxis.pane.set_edgecolor('w')

    # Set plot limits based on 'plot_limit' variable:
    ax2.set_xlim3d(-plot_limit, plot_limit)
    ax2.set_ylim3d(-plot_limit, plot_limit)
    ax2.set_zlim3d(-plot_limit, plot_limit)

    if swap_xy:
        ax2.view_init(30, 50) # view
        text_position = [10, 2, -4]
    else:
        text_position = [0, -10, -4]

    # Text value of the quaternion
    quat_string = 'q='+ str(round(a, 2)) + '+' + str(round(b, 2)) + 'i+' + str(round(c, 2)) + 'j+' + str(round(d, 2)) + 'k'
    t = ax2.text(text_position[0], text_position[1], text_position[2], s=quat_string, fontsize=12, color=text_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    fig.tight_layout(pad=2.0)  # To add padding between subplots.
    plt.show()                 # To show the plot.
    fig.savefig('Quaternion_Figure.png', dpi=400, bbox_inches='tight')   # To save the entire figure.


def view_Euler(yaw_z=0, pitch_y=0, roll_x=0, intrinsic=True, sequence='zyx', radians=False, dark_mode=False, swap_xy=False):
    """
    Input:
        yaw_z: Float value containing the yaw in [deg] or [rad]; rotation about z-axis  
        pitch_y: Float value containing the pitch in [deg] or [rad]; rotation about y-axis    
        roll_x:  Float value containing the roll in [deg] or [rad]; rotation about x-axis
        intrinsic: Boolean value for intrinsic angles: 'True' for intrinsic angles, 'False' for extrinsic angles 
        sequence: String containing the Euler angle rotation sequence; e.g. 'xyz', or 'XYZ'
        radians: Boolean value for whether radians used for yaw, pitch, and roll: 'True' if using radians, 'False' if using degrees.
        dark_mode: Boolean variable for dark mode plotting: 'True' for dark mode plotting; 'False' for white background
        swap_xy: Boolean variable for swapping x-axis and y-axis in 3D view of the sensor
    Output:
        The function renders the figure of your quaternion and saves the image as a .png file.
    """
    # Import Modules:
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.pyplot as plt

    # Lowercase the sequence order
    sequence = sequence.lower()   

    # Ensure sequence is unique (i.e., a Tait-Brian Euler angle) and there are three entries
    if (sequence.count('x') != 1) or (sequence.count('y') != 1) or (sequence.count('z') != 1): 
      print('ERROR: The sequence must be distinct!')
      return # End Function

    # Reset default matplotlib settings:
    plt.rcdefaults()

    # set global parameters for plotting (optional):
    plt.rcParams['figure.figsize'] = [14, 6]
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["font.family"] = "serif"  # Set font style globally to serif (much nicer than default font style).
    plot_limit = 8  # constant for x-axis, y-axis, and z-axis length limits

    # Set dark mode style on if dark_mode == True:
    if dark_mode:  # True if dark_mode == True
      plt.style.use('dark_background')  # set 'dark_background' sylte on
      edge_color = 'white'  # set color for legend box to white
      sensor_color = '#990000' # crimson
      sensor_alpha = 0.6       # alpha transparency between 0 to 1
      sensor_edge_color = 'w'  # white
      x_axis_color = '#ff073a' # neon red
      y_axis_color = '#04d9ff' # neon blue
      z_axis_color = '#39FF14' # neon green
      mag_north_color = 'w'    # white for magnetic north arrow
      text_background_alpha = 0.7
      text_color = 'w'
      text_background = 'k'
    else:         # if dark_mode == False 
      edge_color = 'black'  # set color for legend box to black
      sensor_color = sensor_color = '#990000' # crimson
      sensor_alpha = 0.2
      sensor_edge_color = 'k' # black
      x_axis_color = 'r' # red
      y_axis_color = 'b' # blue
      z_axis_color = 'g' # green
      mag_north_color = 'k'    # white for magnetic north arrow
      text_background_alpha = 0.6
      text_color = 'k'
      text_background = 'w'

    # Define sensor dimensions for plotting:
    width = 4     # y-axis dimension
    height = 1    # z-axis dimension
    depth = 6     # x-axis dimension

    # Define 8 points of rectangular prism:
    pt1 = np.array([-depth/2, -width/2, -height/2 ])
    pt2 = np.array([depth/2, -width/2, -height/2 ])
    pt3 = np.array([depth/2, width/2, -height/2 ])
    pt4 = np.array([-depth/2, width/2, -height/2])
    pt5 = np.array([-depth/2, -width/2, height/2])
    pt6 = np.array([depth/2, -width/2, height/2 ])
    pt7 = np.array([depth/2, width/2, height/2])
    pt8 = np.array([-depth/2, width/2, height/2])

    # define 6 faces of rectangular prism:
    long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
    long_side_face2 = [[pt3, pt7,  pt8, pt4]]
    short_side_face1 = [[pt3, pt2, pt6, pt7]]
    short_side_face2 = [[pt1, pt5, pt8, pt4]]
    bottom_face = [[pt1, pt2, pt3, pt4]]
    top_face = [[pt5, pt6, pt7, pt8]]

    # Define vectors representing coordinate axes x, y, and z:
    arrow_length = depth/2*1.9
    x_axis = np.array([arrow_length, 0, 0])
    y_axis = np.array([0, arrow_length, 0 ])
    z_axis = np.array([ 0, 0, arrow_length ])

    # Define vectors for holding text for each axis:
    x_text = np.array([arrow_length+1, 0, 0])
    y_text = np.array([0, arrow_length+1, 0 ])
    z_text = np.array([ 0, 0, arrow_length+1 ])

    # =============Create Figure 1 of 2: Original sensor orientation================
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot each of the 6 faces of the rectangular prism:
    ax.add_collection3d(Poly3DCollection(long_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(long_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(short_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(short_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(bottom_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax.add_collection3d(Poly3DCollection(top_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))

    # Draw 3 arrows for each local coordinate axis:
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color=x_axis_color, linewidth=2, linestyle='-') # x-axis
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color=y_axis_color, linewidth=2, linestyle='-') # y-axis
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color=z_axis_color, linewidth=2, linestyle='-') # z-axis

    t = ax.text(x_text[0], x_text[1], x_text[2], s='x', color=x_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax.text(y_text[0], y_text[1], y_text[2], s='y', color=y_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax.text(z_text[0], z_text[1], z_text[2], s='z', color=z_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    # Magnetic north and gravity
    if swap_xy:
        start_pt = [6, -6, -6]
        text_position = [6, -5.5 + arrow_length, -6.5]
    else:
        start_pt = [6, -6, -6]
        text_position = [5.5, -4.5 + arrow_length, -6.5]

    # Magnetic north arrow:
    ax.quiver(start_pt[0], start_pt[1], start_pt[2],  0 ,arrow_length, 0, color=mag_north_color, linewidth=1.6, linestyle='-') # z-axis

    # Text for 'N' near the magnetic north arrow:
    t = ax.text(text_position[0], text_position[1], text_position[2], s='N', fontsize=14, color=text_color, fontfamily='serif' , fontweight='bold')
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    # Define axis labels and title:
    ax.set_xlabel('Global X-axis')
    ax.set_ylabel('Global Y-axis')
    ax.set_zlabel('Global Z-axis')
    ax.set_title('Original Orientation' , fontsize='13')

    # Get rid of number on axis:
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Set plot limits based on 'plot_limit' variable:
    ax.set_xlim3d(-plot_limit, plot_limit)
    ax.set_ylim3d(-plot_limit, plot_limit)
    ax.set_zlim3d(-plot_limit, plot_limit)

    if swap_xy:
        ax.view_init(30, 50) # view

    #=====================Rotation==================
    # Define rotation matrix based on yaw, pitch, and yaw:
    R = rot_mat_Euler(yaw_z, pitch_y, roll_x, intrinsic, sequence, radians)

    # ================Rotate all points by rotation matrix R===============
    pt1 = R @ pt1
    pt2 = R @ pt2
    pt3 = R @ pt3
    pt4 = R @ pt4
    pt5 = R @ pt5
    pt6 = R @ pt6
    pt7 = R @ pt7
    pt8 = R @ pt8

    x_axis = R @ x_axis
    y_axis = R @ y_axis
    z_axis = R @ z_axis

    x_text = R @ x_text
    y_text = R @ y_text
    z_text = R @ z_text

    # Redefine faces after rotation:
    long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
    long_side_face2 = [[pt3, pt7,  pt8, pt4]]
    short_side_face1 = [[pt3, pt2, pt6, pt7]]
    short_side_face2 = [[pt1, pt5, pt8, pt4]]
    bottom_face = [[pt1, pt2, pt3, pt4]]
    top_face = [[pt5, pt6, pt7, pt8]]

    # =============Create Figure 2 of 2: Transformed sensor orientation================
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot each of the 6 faces of the rectangular prism:
    ax2.add_collection3d(Poly3DCollection(long_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(long_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(short_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(short_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(bottom_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
    ax2.add_collection3d(Poly3DCollection(top_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))

    # Draw 3 arrows for each local coordinate axis:
    ax2.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color=x_axis_color, linewidth=2, linestyle='-') # x-axis
    ax2.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color=y_axis_color, linewidth=2, linestyle='-') # y-axis
    ax2.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color=z_axis_color, linewidth=2, linestyle='-') # z-axis

    t = ax2.text(x_text[0], x_text[1], x_text[2], s='x', color=x_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax2.text(y_text[0], y_text[1], y_text[2], s='y', color=y_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    t = ax2.text(z_text[0], z_text[1], z_text[2], s='z', color=z_axis_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

    # Define axis labels and title:
    ax2.set_xlabel('Global X-axis')
    ax2.set_ylabel('Global Y-axis')
    ax2.set_zlabel('Global Z-axis')
    ax2.set_title('Transformed Orientation', fontsize='13' )

    # Get rid of number on grids
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.axes.zaxis.set_ticklabels([])

    # Get rid of colored axes planes
    # First remove fill
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax2.xaxis.pane.set_edgecolor('w')
    ax2.yaxis.pane.set_edgecolor('w')
    ax2.zaxis.pane.set_edgecolor('w')

    # Set plot limits based on 'plot_limit' variable:
    ax2.set_xlim3d(-plot_limit, plot_limit)
    ax2.set_ylim3d(-plot_limit, plot_limit)
    ax2.set_zlim3d(-plot_limit, plot_limit)

    if swap_xy:
        ax2.view_init(30, 50) # view
        text_position = [9, 5, -6]
    else:
        text_position = [3, -10, -6]

    # Text value of the Euler angles:
    if intrinsic == True:
      intrinsic_str = 'intrinsic'
    else:
      intrinsic_str = 'extrinsic'
    if radians == False:
      Euler_angle_string = intrinsic_str + ': ' + sequence + ',\nyaw_z: ' + str(round(yaw_z, 2)) + '$^{\circ}$,\npitch_y: ' + str(round(pitch_y, 2)) + '$^{\circ}$,\nroll_x: ' + str(round(roll_x, 2)) + '$^{\circ}$'
    else:
      Euler_angle_string = intrinsic_str + ': ' + sequence + ',\nyaw_z: ' + str(round(yaw_z, 2)) + ' rad.,\npitch_y: ' + str(round(pitch_y, 2)) + ' rad.,\nroll_x:' + str(round(roll_x, 2)) + ' rad.'
    t = ax2.text(text_position[0], text_position[1], text_position[2], s=Euler_angle_string, fontsize=11, color=text_color, fontfamily='serif', backgroundcolor=text_background)
    t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
    

    fig.tight_layout(pad=2.0)  # To add padding between subplots.
    plt.show()                 # To show the plot.
    fig.savefig('Euler_angle_Figure.png', dpi=400, bbox_inches='tight')   # To save the entire figure.


def view_orientation(file_path, dark_mode=False, view_Euler=False, view_quat=False, swap_xy=False, num_frame=50, frame_delay=200):
    """
    Input:
        file_path: string containing the file path location (or URL link) of your .csv.
        dark_mode: Boolean variable for dark mode plotting: 'True' for dark mode plotting; 'False' for white background.
        view_Euler: Boolean variable for showing plot of Euler angles animate alongside the orientation animation
        view_quat: Boolean variable for showing plot of quaternions animate alongside the orientation animation 
        swap_xy: Boolean variable for swapping x-axis and y-axis in 3D view of the sensor.
        num_frame: Integer value representing the number of frames you wish to show. 
        frame_delay: Integer value representing the delay between frames in milliseconds.
    Output:
        The function renders a gif of your sensors orientation for visualization and saves the .gif file.
    """

    # Import Libraries:
    import pandas as pd                 # Python library for data manipulation and analysis
    import matplotlib.pyplot as plt     # Python library for data visualization
    import numpy as np                  # Numerical Python 
    from matplotlib import animation    # Matplotlib library for animatin
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # Library for 3D rendering.
    from IPython.display import Image   # For rendering gif on notebook.

    # Call prep_data() to prep and tidy the data:
    data_a, data_g, data_l, data_m, data_r = prep_data(file_path)


    #===========Adjust data_r to have size of 'num_frame'====================
    # Make a copy of data_r:
    data_r2 = data_r.copy()

    # ==============Add datetime to data_r2:
    # Step 1: Enter the data collection date. 
    deploy_date = '2022-09-08 09:50:00'  # Can be anything

    # Step 2: Convert `deploy_date` to a datetime object:
    deploy_date = pd.to_datetime(deploy_date)

    # Step 3: Convert `date.elapsed_time` to datetime.  Save it under a new series `new_time`
    new_time = pd.to_datetime(data_r2.time, unit='s')

    # Step 4: Adjust the date. 
        # Compute timedelta between `deploy_date` and the first recorded time in `new_time`
    time_diff = deploy_date-new_time[0]  # This creates a timedelta
        # Add the `time_diff` to `new_time` to adjust the date.
    new_time = new_time+time_diff

    # Step 5: Store `new_time` in new variable 'date':
    data_r2['date'] = new_time

    # ============= Time Resampling
    # Resample based on 'num_frame'
    num_index = data_r2.index[-1]
    dt = round(data_r2.loc[num_index, 'time']/num_frame,2)
    time_string = str(dt)+'S'
    data_r2 = data_r2.resample(time_string, on='date').mean()

    # Reset index:
    data_r2.reset_index(inplace=True)

    #================ Let's fix time
    # Compute the time delta between datetimes using .diff()
    t_delta = data_r2.date.diff()

    # Convert time delta into 'seconds'
    t_delta = t_delta.dt.total_seconds()

    # Fill NaN with zero
    t_delta = t_delta.fillna(0)

    # Take cumulutive sum and make new variable.
    data_r2['time'] = t_delta.cumsum()

    #=========================Code to create animation==================
    # Reset default matplotlib settings:
    plt.rcdefaults()

    # set global parameters for plotting (optional):
    #plt.rcParams['figure.figsize'] = [10, 4]
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "serif"  # Set font style globally to serif (much nicer than default font style).
    plot_limit = 8  # constant for x-axis, y-axis, and z-axis length limits

    # Set dark mode style on if dark_mode == True:
    if dark_mode:  # True if dark_mode == True
        plt.style.use('dark_background')  # set 'dark_background' sylte on
        edge_color = 'white'  # set color for legend box to white
        sensor_color = '#990000' # crimson
        sensor_alpha = 0.6       # alpha transparency between 0 to 1
        sensor_edge_color = 'w'  # white
        mag_north_color = 'w'    # white for magnetic north arrow
        x_axis_color = '#ff073a' # neon red
        y_axis_color = '#04d9ff' # neon blue
        z_axis_color = '#39FF14' # neon green
        text_background_alpha = 0.7
        text_color = 'w'
        text_background = 'k'
    else:         # if dark_mode == False 
        edge_color = 'black'  # set color for legend box to black
        sensor_color = sensor_color = '#990000' # crimson
        sensor_alpha = 0.2
        sensor_edge_color = 'k' # black
        mag_north_color = 'k'    # white for magnetic north arrow
        x_axis_color = 'r' # red
        y_axis_color = 'b' # blue
        z_axis_color = 'g' # green
        text_background_alpha = 0.6
        text_color = 'k'
        text_background = 'w'

    # Define sensor dimensions for plotting:
    width = 4     # y-axis dimension
    height = 1    # z-axis dimension
    depth = 6     # x-axis dimension


     # Determine number of rows and columns of plots, based on the sensor configuration:
    if not(view_quat) and not(view_Euler):  # Both view_quat and view_Euler are false, only one figure 
        num_rows = 1
        num_cols = 1
        fig_size = (8, 4) # Figure size in inches (default units)
    elif (view_quat and not(view_Euler)) or (view_Euler and not(view_quat)):  # Two figures
        num_rows = 2
        num_cols = 1
        fig_size=(8,6) # Figure size in inches (default units)
    elif (view_quat and view_Euler):  # Three Figures
        num_rows = 3
        num_cols = 1
        fig_size=(8,12) # Figure size in inches (default units)

    # Set size globally:
    plt.rcParams['figure.figsize'] = fig_size

   
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(num_rows, num_cols, 1, projection="3d")  # First Plot is always the orientation animation
    if swap_xy:
        ax.view_init(30, 50) # view

    if num_rows == 2:
        ax2 = fig.add_subplot(num_rows, num_cols, 2)   # Add second figure
        #ax2 = fig.add_subplot(num_rows, num_cols, 2, aspect=0.4, anchor='N')   # Add second figure

    elif num_rows == 3:
        ax2 = fig.add_subplot(num_rows, num_cols, 2)   # Add second figure
        ax3 = fig.add_subplot(num_rows, num_cols, 3)   # Add third figure

    #fig.tight_layout(pad=2.0)  # To add padding between subplots.


    # Update plot function for animation:
    def update(frame):
        ax.clear()

        # Define 8 points of rectangular prism:
        pt1 = np.array([-depth/2, -width/2, -height/2 ])
        pt2 = np.array([depth/2, -width/2, -height/2 ])
        pt3 = np.array([depth/2, width/2, -height/2 ])
        pt4 = np.array([-depth/2, width/2, -height/2])
        pt5 = np.array([-depth/2, -width/2, height/2])
        pt6 = np.array([depth/2, -width/2, height/2 ])
        pt7 = np.array([depth/2, width/2, height/2])
        pt8 = np.array([-depth/2, width/2, height/2])

        # define 6 faces of rectangular prism:
        long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
        long_side_face2 = [[pt3, pt7,  pt8, pt4]]
        short_side_face1 = [[pt3, pt2, pt6, pt7]]
        short_side_face2 = [[pt1, pt5, pt8, pt4]]
        bottom_face = [[pt1, pt2, pt3, pt4]]
        top_face = [[pt5, pt6, pt7, pt8]]

        # Define vectors representing coordinate axes x, y, and z:
        arrow_length = depth/2*1.9
        x_axis = np.array([arrow_length, 0, 0])
        y_axis = np.array([0, arrow_length, 0 ])
        z_axis = np.array([ 0, 0, arrow_length ])

        # Define vectors for holding text for each axis:
        x_text = np.array([arrow_length+1.1, 0, 0])
        y_text = np.array([0, arrow_length+1.1, 0 ])
        z_text = np.array([ 0, 0, arrow_length+1.1 ])

        # Define quaternion values for current frame:
        a = data_r2.loc[frame, 'a']
        b = data_r2.loc[frame, 'b']
        c = data_r2.loc[frame, 'c']
        d = data_r2.loc[frame, 'd']

        # Normalize quaternion (Rotation matrix needs unit quaternion)
        mag = np.sqrt(a**2 + b**2 + c**2 + d**2)
        if mag == 0.0:
          print("ERROR: Zero quaternion encoutered.")
          return # End Function
        a = a/mag
        b = b/mag
        c = c/mag
        d = d/mag

        # Define rotation matrix based on quaternion values a, b, c, and d:
        R = rot_mat(a, b, c, d)

        # Rotate all points by rotation matrix R
        pt1 = R @ pt1
        pt2 = R @ pt2
        pt3 = R @ pt3
        pt4 = R @ pt4
        pt5 = R @ pt5
        pt6 = R @ pt6
        pt7 = R @ pt7
        pt8 = R @ pt8
        x_axis = R @ x_axis
        y_axis = R @ y_axis
        z_axis = R @ z_axis
        x_text = R @ x_text
        y_text = R @ y_text
        z_text = R @ z_text

        # Redefine faces after rotation:
        long_side_face1 = [[ pt1, pt5, pt6, pt2 ]]
        long_side_face2 = [[pt3, pt7,  pt8, pt4]]
        short_side_face1 = [[pt3, pt2, pt6, pt7]]
        short_side_face2 = [[pt1, pt5, pt8, pt4]]
        bottom_face = [[pt1, pt2, pt3, pt4]]
        top_face = [[pt5, pt6, pt7, pt8]]

        # Plot each of the 6 faces of the rectangular prism:
        ax.add_collection3d(Poly3DCollection(long_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
        ax.add_collection3d(Poly3DCollection(long_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
        ax.add_collection3d(Poly3DCollection(short_side_face1, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
        ax.add_collection3d(Poly3DCollection(short_side_face2, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
        ax.add_collection3d(Poly3DCollection(bottom_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))
        ax.add_collection3d(Poly3DCollection(top_face, facecolors=sensor_color, linewidths=1., edgecolors=sensor_edge_color, alpha=sensor_alpha))

        # Draw 3 arrows for each local coordinate axis:
        ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color=x_axis_color, linewidth=2, linestyle='-') # x-axis
        ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color=y_axis_color, linewidth=2, linestyle='-') # y-axis
        ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color=z_axis_color, linewidth=2, linestyle='-') # z-axis

        t = ax.text(x_text[0], x_text[1], x_text[2], s='x', color=x_axis_color, fontfamily='serif', backgroundcolor=text_background)
        t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
        t = ax.text(y_text[0], y_text[1], y_text[2], s='y', color=y_axis_color, fontfamily='serif', backgroundcolor=text_background)
        t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))
        t = ax.text(z_text[0], z_text[1], z_text[2], s='z', color=z_axis_color, fontfamily='serif', backgroundcolor=text_background)
        t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))

        # Magnetic north and gravity
        if swap_xy:
            start_pt = [6, -6, -6]
            text_position = [6, -5 + arrow_length, -6.5]
        else:
            start_pt = [6, -6, -6]
            text_position = [6.5, -5.5 + arrow_length, -6.5]

        # Magnetic north arrow:
        ax.quiver(start_pt[0], start_pt[1], start_pt[2],  0 ,arrow_length, 0, color=mag_north_color, linewidth=1.6, linestyle='-') # z-axis

        # Text for 'N' near the magnetic north arrow:
        t= ax.text(text_position[0], text_position[1], text_position[2], s='N', fontsize=14, color=text_color, fontfamily='serif' , fontweight='bold')
        t.set_bbox(dict(facecolor=text_background, alpha=text_background_alpha, linewidth=0))


        # Define axis labels and title:
        ax.set_xlabel('Global X-axis')
        ax.set_ylabel('Global Y-axis')
        ax.set_zlabel('Global Z-axis')
        ax.set_title('Time = ' + str(np.round(data_r2.loc[frame, 'time'],decimals=2)) + ' sec')

        # Get rid of number on grids
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Set plot limits based on 'plot_limit' variable:
        ax.set_xlim3d(-plot_limit, plot_limit)
        ax.set_ylim3d(-plot_limit, plot_limit)
        ax.set_zlim3d(-plot_limit, plot_limit)


        # Second Figure:
        if ((num_rows == 2) and (view_Euler)):
            ax2.clear()
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'roll_x'],'-', c = x_axis_color  ,label="Roll")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'pitch_y'], '-', c = y_axis_color , label="Pitch")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'yaw_z'], '-', c =  z_axis_color, label="Yaw")
            ax2.legend(ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')

            # Find max for setting ylim:
            max_y = max(data_r2.roll_x.max(), data_r.pitch_y.max(), data_r.yaw_z.max())
            min_y = min(data_r2.roll_x.min(), data_r.pitch_y.min(), data_r.yaw_z.min())

            ax2.set_xlim(0, data_r2.time.max()*1.05)
            ax2.set_ylim(min_y, max_y*1.05)
            ax2.tick_params(which='major', width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax2.tick_params(which='minor', width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel("Euler Angles  [degrees $^{\circ}$]")
            ax2.minorticks_on()  # Turns minor thicks on (optional)
            ax2.grid()           # Shows grid
        elif ((num_rows == 2) and (view_quat)):
            ax2.clear()
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'a'],'-', c = x_axis_color  ,label="a")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'b'], '-', c = y_axis_color , label="b")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'c'], '-', c =  z_axis_color, label="c")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'd'], '-', c =  'y', label="d")
            ax2.legend(ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')

            # Find max for setting ylim:
            max_y = max(data_r2.a.max(), data_r.b.max(), data_r.c.max() , data_r.d.max() )
            min_y = min(data_r2.a.min(), data_r.b.min(), data_r.c.min() , data_r.d.min())

            ax2.set_xlim(0, data_r2.time.max()*1.05)
            ax2.set_ylim(min_y, max_y*1.05)
            ax2.tick_params(which='major', width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax2.tick_params(which='minor', width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel("Value [-]")
            ax2.minorticks_on()  # Turns minor thicks on (optional)
            ax2.grid()           # Shows grid


        if (num_rows == 3):

            # Second Figure
            ax2.clear()
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'roll_x'],'-', c = x_axis_color  ,label="Roll")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'pitch_y'], '-', c = y_axis_color , label="Pitch")
            ax2.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'yaw_z'], '-', c =  z_axis_color, label="Yaw")
            ax2.legend(ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')

            # Find max for setting ylim:
            max_y = max(data_r2.roll_x.max(), data_r.pitch_y.max(), data_r.yaw_z.max())
            min_y = min(data_r2.roll_x.min(), data_r.pitch_y.min(), data_r.yaw_z.min())

            ax2.set_xlim(0, data_r2.time.max()*1.05)
            ax2.set_ylim(min_y, max_y*1.05)
            ax2.tick_params(which='major', width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax2.tick_params(which='minor', width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel("Euler Angles  [degrees $^{\circ}$]")
            ax2.minorticks_on()  # Turns minor thicks on (optional)
            ax2.grid()           # Shows grid
       

            # Third Figure:
            ax3.clear()
            ax3.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'a'],'-', c = x_axis_color  ,label="a")
            ax3.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'b'], '-', c = y_axis_color , label="b")
            ax3.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'c'], '-', c =  z_axis_color, label="c")
            ax3.plot(data_r2.loc[:frame, 'time'], data_r2.loc[:frame, 'd'], '-', c =  'y', label="d")
            ax3.legend(ncol=1, framealpha=0.5, fancybox=False, edgecolor=edge_color, loc='best')

            # Find max for setting ylim:
            max_y = max(data_r2.a.max(), data_r.b.max(), data_r.c.max() , data_r.d.max() )
            min_y = min(data_r2.a.min(), data_r.b.min(), data_r.c.min() , data_r.d.min())

            ax3.set_xlim(0, data_r2.time.max()*1.05)
            ax3.set_ylim(min_y, max_y*1.05)
            ax3.tick_params(which='major', width=1, length=7,  direction='in', bottom = True, top= True, left= True, right= True) # set major thicks (optional)
            ax3.tick_params(which='minor', width=1, length=3,  direction='in', bottom = True, top= True, left= True, right= True) # set minor thicks (optional)
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel("Value [-]")
            ax3.minorticks_on()  # Turns minor thicks on (optional)
            ax3.grid()           # Shows grid



    # Define the Animation
    ani = animation.FuncAnimation(fig, update, frames=num_frame, interval=frame_delay)
    plt.close()

    # Saving the Animation
    file_name = file_path.split('/')[-1][:-4] + '_animation.gif'
    f = '/content/' + file_name  # This only works in Google Colab!!
    writergif = animation.PillowWriter(fps=num_frame/8)
    ani.save(f, writer=writergif)

    # Render gif on Google Colab notebook:
    return Image(open(file_name,'rb').read())
