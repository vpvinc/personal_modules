"""Useful fonctions and params"""

print("""
# plot params
- set_style_pers

# EDA functions
 ## multiple dfs exploration
- dfs_insight
 ## univariate analysis
- plot_cont_kde
- plot_cat_countplot
- barPerc
# Clustering functions
- radar_ploting_clustering
""")

# plot params

def set_style_pers():
    """set default sns style and customize rc params"""
    
    from cycler import cycler
    matplotlib.rcParams.update(
        {    
        'axes.titlesize'     :25,                 # axe title
        'axes.labelsize'     :20,
        'axes.prop_cycle'    :cycler('color', ['#0D8295', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),          # lines colors
        'lines.linewidth'    :3,
        'lines.markersize'   :150,
        'xtick.labelsize'    :15,
        'ytick.labelsize'    :15,
        'font.family'        :'Century Gothic'
        }
        )

# class to easily print text in color or with effects

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# EDA functions

def dfs_insight(df_dict):
    """ give general information on dfs of a list: shape, columns, % of NaN, duplicates
            -input: a dict of dataframes with their name as key
            
            -output: 
                1) print a df with df-wise information: name_df, shape, % NaN, # duplicated rows
                2) print a df per df in the input dict: each printed df contains column-wise info: 
                # dupplicated entries, avg_nan rate per column
                """
    from IPython.core.display import display
    
    # 1 df that will contain all df-wise info of the different dfs in the input dict
    dfs_display = pd.DataFrame(columns=['name_df', 'shape', '% NaN', '# duplicated rows'])
    # 1 dict to keep the dfs with column_wise info
    df_cols_display_dict = {}
    
    for name, df in df_dict.items():
        # compute general indicators on each df
        shape = df.shape,
        avg_nan = '{:.2%}'.format(df.isna().mean().mean())
        nb_dupli_rows = df.duplicated().value_counts().filter([True]).values
        if not nb_dupli_rows.size > 0:
            nb_dupli_rows = 0
        else:
            nb_dupli_rows = int(nb_dupli_rows)
        # make a series with these indicators and add the indicators names as index 
        df_display = pd.Series([name, shape, avg_nan, nb_dupli_rows], index = dfs_display.columns)
        # add this series as a row to the general df that will display info about all dfs
        dfs_display = dfs_display.append(df_display, ignore_index=True)
        
        # making a df for columns indicators (1 df per df), the column of the described df are the index
        df_cols_display = pd.DataFrame(index=df.columns)
        # adding the dtype
        df_cols_display['dtype']= df.dtypes
        # computing the number of dupplicated entries per column
        try:
            nb_dupli_entries = df.apply(lambda col: col.duplicated().value_counts()).loc[True]
            # if there is no dupplicated entries, we get NaN, we have to replace it by 0
            nb_dupli_entries.fillna(0,inplace=True)
            df_cols_display['# dupplicated entries']  = nb_dupli_entries.values
        except KeyError:
            nb_dupli_entries = 0
            df_cols_display['# dupplicated entries']  = nb_dupli_entries
         
        # computing the avg of NaN per column    
        avg_nan_col = df.isna().mean()
        avg_nan_col = pd.Series(['{:.2%}'.format(val) for val in avg_nan_col], index=avg_nan_col.index)
        df_cols_display['avg_nan_col'] = avg_nan_col.values

        # adding the df to the dict for further displaying 
        df_cols_display_dict[name] = df_cols_display
        
    
    # printig the general df with df-wise info
    print(color.BOLD,color.UNDERLINE, 'df-wise information', color.END)
    print(dfs_display, '\n')
    # printing 1 df per df with column-wise-info
    print(color.BOLD,color.UNDERLINE, 'column-wise information', color.END)
    for name, df in df_cols_display_dict.items():
        print(color.BOLD, name, color.END)
        print(df)
        print('\nfirst 3 rows\n')
        display(df_dict[name].head(3))
        print('-'*70)

def plot_cont_kde(data_df, var_list, hue=None, w=30,h=6):
    """# plot kde plot with median and Std values for one or more designated variables
    
    args: 
    - df 
    - [variables] to plot (list with just the names between quotes, without the df)
    WIP: 
    - hue: (str) key in df plugged into sns.kdeplot: Semantic variable that is mapped to determine the color of plot elements. 
    ->  legend does not show as in classic sns.kdeplot 
    
    kwargs:
    - width
    - height
    
    returns:
    - 1 boxplot
    - 1 kde plot with markers for min, max, med, mean
    - title with range, kurtosis and skew
    """
    # seting up the fig and the number of subplots according to the number of variables
    fig, axes=plt.subplots(len(var_list),2)
    fig.set_size_inches(w,h*len(var_list))
    # setting up the space between subplots
    fig.tight_layout(h_pad=8, w_pad=3)
    
    # plotting for each variable
    for i, var in enumerate(var_list):
    
        # computing indicators to display
        mini = data_df[var].min()
        maxi = data_df[var].max()
        ran = data_df[var].max()-data_df[var].min()
        mean = data_df[var].mean()
        skew = data_df[var].skew()
        kurt = data_df[var].kurtosis()
        median = data_df[var].median()
        st_dev = data_df[var].std()
        points = mean-st_dev, mean+st_dev
        
        # boxplot
        sns.boxplot(x=data_df[var],
                    ax=axes[i, 0],
                    color='#0D8295',
                    medianprops={'color':'white', 'linewidth':3 },
                    showmeans=True, 
                    meanprops={"marker":"o",
                                "markerfacecolor":"red", 
                                "markeredgecolor":"white",
                                "markersize":"15"})
        axes[i, 0].set_xlabel(var, fontsize=25, **{'fontname':'Century Gothic'})
        axes[i, 0].tick_params(axis='both', labelsize=15)
        
        # kde plot
        sns.kdeplot(data=data_df, x=var, ax=axes[i, 1], color='#0D8295', hue=hue)
        axes[i, 1].set_ylabel(None)
        
        # ploting indicators over the kde plot
        max_y = axes[i, 1].get_ylim()[1] # this enables us to plot indicators at a readable scale VS kde plot, see below
        sns.lineplot(x=points, y=[max_y/2.5,max_y/2.5], ax=axes[i, 1], linestyle='dashed', color = 'black', label = "std_dev")
        sns.scatterplot(x=mini, y=[max_y/2], ax=axes[i, 1], color = 'orange', label = "min", marker='>', s=200)
        sns.scatterplot(x=maxi, y=[max_y/2], ax=axes[i, 1], color = 'orange', label = "max", marker='<', s=200)
        sns.scatterplot(x=[mean], y=[max_y/2], ax=axes[i, 1], color = 'red', label = "mean", marker='o', s=150)
        sns.scatterplot(x=[median], y=max_y/2, ax=axes[i, 1], color = 'blue', label = "median", marker='v', s=150)
        
        # setting tile, ticks and displaying indicators over the kde plot
        axes[i, 1].set_xlabel(var, fontsize=25,**{'fontname':'Century Gothic'})
        axes[i, 1].tick_params(axis='both', labelsize=15)
        axes[i, 1].set_title('std_dev = {} || kurtosis = {} || nskew = {} || range = {} \n || nmean = {} || median = {}'.format((round(points[0],2), 
                round(points[1],2)),
                round(kurt,2),round(skew,2),(round(mini,2),round(maxi,2),
                round(ran,2)),round(mean,2), round(median,2)), fontsize=20)
        axes[i, 1].legend(fontsize=15)
        
def plot_cat_countplot(data, var_list, index_col, w=30, h=6):
    """# plot countplot for one or more designated variables
    
    args: 
    - df 
    - [variables] to plot (list with just the names between quotes, without the df)
    - name of column used as index for counting (must be a string) ex: 'customer_id'
    
    kwargs:
    - width
    - height
    
    returns:
    - 1 countplot with modalities on X-axis and count on Y-axis
    """
    fig, axes = plt.subplots(ceil(len(var_list)/2),2)
    fig.set_size_inches(w,h*len(var_list)*0.8)
    fig.tight_layout(w_pad=6, h_pad=6)
    for axe, ind  in enumerate(data[var_list]):
        data_count = data.groupby(ind)[index_col].count()
        ax = sns.barplot(x=data_count.values, y=data_count.index, palette='YlGnBu', ax = axes.ravel()[axe], orient='h')
        ax.set_ylabel(ind, fontsize=30)
        ax.set_xlabel('count {}'.format(index_col), fontsize=30)
        # ax.tick_params(rotation=45)
        
def barPerc(df,xVar,ax):
    '''
    barPerc(): Add percentage for hues to bar plots
    args:
        df: pandas dataframe
        xVar: (string) X variable 
        ax: Axes object (for Seaborn Countplot/Bar plot or
                         pandas bar plot, hue must be specified
                         a parameter of the plot)
    '''
    # 1. how many X categories
    ##   check for NaN and remove
    numX=len([x for x in df[xVar].unique() if x==x])

    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    ## 2a. For each X variable
    for ind in range(numX):
        ## 2b. Get every hue bar
        ##     ex. 8 X categories, 4 hues =>
        ##    [0, 8, 16, 24] are hue bars for 1st X category
        hueBars=bars[ind:][::numX]
        ## 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hueBars])

        # 3. Print the percentage on the bars
        for bar in hueBars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height(),
                    f'{bar.get_height()/total:.0%}',
                    ha="center",va="bottom")
        
# Clustering functions

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def radar_ploting_clustering(df, title):
    """DEPENDENT to function radar_factory in the same module useful_functions.py
    
    Plot a radar chart displaying the different mean values of clusters for the numerical features of the df. Data should be standardized.
    The df should be grouped by clusters, and the clusters must be indicated in the first column of the df with name 'cluster'
    
    example of df:
         cluster  |  nb_orders  |  review_score  |  review_delay
   -----------------------------------------------------------
    0      0      |  1.000000   |  4.501408      |  9.215493
    1      1      |  1.000000   |  4.429864      |  6.090498
    2      2      |  2.145161   |  4.064516      |  5.500000
    
    parameters:
    - dataframe with the cluster indicated in the first column as 'cluster', other columns are numerical features that will form the axes of the radar
    - title to be given to the plot
    
    Output
    - radar chart with legend"""
    
    N = len(df.iloc[:, 1:].columns)
    theta = radar_factory(N, frame='polygon')
    
    clusters = df.iloc[:, 0].values
    spoke_labels = df.iloc[:, 1:].columns
    case_data = df.iloc[:, 1:].values

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([]) # remove r ticks
    ax.set_title(title,  position=(0.5, 1.), ha='center', fontsize=25)

    for d in case_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d,  alpha=0.15)
    ax.set_varlabels(spoke_labels)
    
    # add legend relative to top-left plot    
    fig.text(0.98, 0.84, 'Clusters', fontsize=14)
    legend = plt.gca().legend(clusters, loc=(1.1, .80),                               
                      labelspacing=0.1, fontsize=12)

    plt.show()