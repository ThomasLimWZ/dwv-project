import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
from IPython.display import display
import missingno as msno


# from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Plot for dataframe
def display_df_details(df):
    # Gather the required information
    summary_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })

    # Create the 'Unique Values List' column
    unique_values_list = []
    for col in df.columns:
        if df[col].dtype == 'float':
            min_val = df[col].min()
            max_val = df[col].max()
            unique_values_list.append(f"Range: {min_val} - {max_val}")
        else:
            unique_values_list.append(df[col].dropna().unique())

    summary_df['Values List'] = unique_values_list

    # Apply the styling directly using a lambda function
    styled_df = (summary_df.style
        .apply(lambda s: ['color: red' if v > 0 else '' for v in s], subset=['Null Count'])
        # Chain the next styling for 'Dtype' column where the Dtype is 'object'
        .apply(lambda s: ['color: green' if str(v) == 'object' else '' for v in s], subset=['Dtype'])
        .set_properties(**{'text-align': 'left'})
    )

    # Display the styled DataFrame
    display(styled_df)

def encode_objectdtypes_columns(df, label_encode_columns):
    for column in df.columns:
        # Converting all to lowercase when it is string data type
        unique_values_lower = set(x.lower() if isinstance(x, str) else x for x in df[column].unique())
        # Apply mapping for the specific column
        if unique_values_lower == {"yes", "no"}:
            df[column] = df[column].str.lower().map({"yes": 1, "no": 0})

    # Label encoding columns using LabelEncoder()
    label_encoder = LabelEncoder()
    for column in label_encode_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df

# ******************************************* Data Exploration Funtions ****************************************************
def missing_values_bar_chart(df):
    # Creating a figure and axis object with specified size and resolution
    fig, ax = plt.subplots(figsize=(14, 5), dpi=70)
    fig.patch.set_facecolor('#f6f5f5')  # Setting background color for the figure
    ax.set_facecolor('#f6f5f5')  # Setting background color for the axis

    # Creating a missing value bar chart for the dataframe `df`
    # Sorting the bars in descending order, setting custom colors, and other styling options
    msno.bar(df, sort='descending', color=['grey']*11 + ['#fe346e'], ax=ax, fontsize=9, labels='off', filter='top')

    # Adding a title text above the plot
    ax.text(-1, 1.35, 'Visualisation of Nullity of the Dataset', 
            {'font': 'Serif', 'size': 24, 'color': 'black'}, alpha=0.9)

    # Adding a subtitle text with description below the title
    ax.text(-1, 1.2, 'Overall there are 5110 datapoints present in \nthe given dataset. Only "bmi" variable has null values.', 
            {'font': 'Serif', 'size': 12, 'color': 'black'}, alpha=0.7)

    # Customizing x-tick labels with specific font properties and rotating them for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', 
                    **{'font': 'Serif', 'size': 14, 'weight': 'normal', 'color': '#512b58'}, alpha=1)

    # Removing y-tick labels
    ax.set_yticklabels('')

    # Making the bottom spine visible
    ax.spines['bottom'].set_visible(True)

    # Displaying the plot
    plt.show()

# Plot to check is data is balanced
# Function to visualise categorical data with pie and bar charts
def categorical_data_visualisation(df, cat_columns, title, subtitle):
    categorical_columns = ['gender', 'work_type', 'hypertension', 'heart_disease', 'ever_married', 'smoking_status', 'Residence_type']

    # Check if specified columns exist in the DataFrame
    missing_columns = set(categorical_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    # Check if the list of categorical columns is empty
    if not categorical_columns:
        raise ValueError("List of categorical columns is empty.")

    df_eda = df.copy()  # Create a copy of the DataFrame to avoid modifying the original one
    for column in categorical_columns:
        if df_eda[column].isin([0, 1]).all():
            df_eda[column] = df_eda[column].map({0: 'No', 1: 'Yes'})

    # Setting up the figure with custom dimensions and DPI
    fig = plt.figure(figsize=(25, 20), dpi=40)

    # Creating a 3x4 grid for subplots (added an extra column for the pie chart)
    gs = fig.add_gridspec(3, 5)
    gs.update(wspace=0.2, hspace=0.6)

    # Defining subplots for the specific display order
    ax_pie = fig.add_subplot(gs[:, :2])  # Pie chart will occupy the first column entirely
    ax1 = fig.add_subplot(gs[0, 2])  # Gender
    ax2 = fig.add_subplot(gs[0, 3:])  # Work Type
    ax3 = fig.add_subplot(gs[1, 2])  # Hypertension
    ax4 = fig.add_subplot(gs[1, 3])  # Heart Disease
    ax5 = fig.add_subplot(gs[1, 4])  # Marital Status
    ax6 = fig.add_subplot(gs[2, 2:4])  # Smoking Status
    ax7 = fig.add_subplot(gs[2, 4])  # Residence Type

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    # gs.update(wspace=0.3, hspace=0.8)  # Reduce wspace to give more room for the pie chart

    # Set figure background color
    fig.patch.set_facecolor('#f5f5f5')

    # Text formatting dictionaries for title, labels, etc.
    title_args = {'font': 'Serif', 'weight': 'bold', 'color': 'black', 'size': 24}
    font_dict = {'size': 16, 'family': 'Serif', 'color': 'black', 'weight': 'bold'}

    # Custom colors 
    stroke_col = '#fe346e'  # For Stroke Group
    healthy_col = '#2c003e'  # For Healthy Group

    # Plotting the pie chart on the left (first column)
    x = df_eda['stroke'].value_counts()

    # Pie chart
    colors = ['#512b58', '#fe346e']
    labels = ['Healthy', 'Stroke']
    _, _, autotexts = ax_pie.pie(
        x, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 24, 'fontweight': 'bold', 'fontfamily': 'Serif'},
        pctdistance=0.85, labeldistance=1.1
    )

    # Set the face color for the pie chart
    ax_pie.set_facecolor('#f5f5f5')  # Set the background for the pie chart axis

    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(21)
        autotext.set_weight('bold')

    # Set title for the pie chart
    ax_pie.set_title('Stroke Distribution', fontweight='bold', fontsize=24, fontfamily='Serif', x=0.4, y=1.55)
    ax_pie.text(-1.2, 1.25, subtitle, ha='left', fontsize=21, fontfamily='Serif')

    # Plotting bar charts for each categorical variable
    for i, (ax, column) in enumerate(zip(axes, categorical_columns)):
        healthy_gen = df_eda[df_eda['stroke'] == 0][column].value_counts()
        stroke_gen = df_eda[df_eda['stroke'] == 1][column].value_counts()

        if column == 'gender':
            # Plot a horizontal bar chart for gender
            ax.barh(healthy_gen.index, width=healthy_gen.values, height=0.2, color=healthy_col)
            ax.barh(np.arange(len(stroke_gen.index)), width=stroke_gen.values, height=0.5, color=stroke_col)
            
            ax.set_yticks(np.arange(len(stroke_gen.index)))
            ax.set_yticklabels(stroke_gen.index, **font_dict)
        else:
            # Plot normal bar charts for other categories
            ax.bar(healthy_gen.index, height=healthy_gen.values, width=0.2, color=healthy_col)
            ax.bar(np.arange(len(stroke_gen.index)), height=stroke_gen.values, width=0.4, color=stroke_col)
            ax.set_xticks(np.arange(len(healthy_gen.index)))
            ax.set_xticklabels(healthy_gen.index, **font_dict)

        ax.set_title(f'{column.capitalize()}', **title_args)

        # Customize axes appearance
        ax.axes.get_yaxis().set_visible(False)  # Hide y-axis for all but the horizontal bar chart
        ax.set_facecolor('#f5f5f5')  # Set background color for each axis
        ax.spines['bottom'].set_linewidth(2)  # Bottom spine thicker
        for loc in ['left', 'right', 'top']:  # Hide other spines
            ax.spines[loc].set_visible(False)
            ax.spines[loc].set_linewidth(2)
    
        # Adding legend and annotations
        ax.legend(['Healthy', 'Stroke'], loc='upper right', prop={'family': 'Serif', 'weight': 'bold', 'size': 12})
    ax1.spines['left'].set_visible(True)
    ax1.get_yaxis().set_visible(True)
    ax1.get_xaxis().set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Title for the entire figure
    fig.suptitle(title, fontweight='bold', fontsize=24, y=0.95, fontfamily='Serif')

    # Show the plot
    plt.show()
    
# Correlation Heatmap
def plot_correlation_heatmap(df):
    colors = ['#f6f5f5','#512b58','#fe346e']
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    corr = sns.clustermap(df.corr(), annot=True, fmt='0.2f',
                    cbar=False, cbar_pos=(0, 0, 0, 0), linewidth=0.5,
                    cmap=colormap, dendrogram_ratio=0.1,
                    facecolor='#f6f5f5', figsize=(8, 8),
                    annot_kws={'font': 'serif', 'size': 10, 'color': 'black'})

    plt.gcf().set_facecolor('#f6f5f5')
    label_args = {'font': 'serif', 'font': 18, 'weight': 'bold'}
    plt.setp(corr.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=10, fontfamily='Serif', fontweight='bold', alpha=0.8)  # For y-axis
    plt.setp(corr.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=10, fontfamily='Serif', fontweight='bold', alpha=0.8)  # For x-axis
    corr.fig.text(0, 1.065, 'Visualisation of Clustering of Each Variable with Other', {'font': 'serif', 'size': 16, 'weight': 'bold'})
    corr.fig.text(0, 1.015, 'Lines on the top and left of the cluster map are called \ndendrograms, which indicate the dependency of variables.', {'font': 'serif', 'size': 12}, alpha=0.8)
    plt.show()
    
# Correlation Pair Plot
def cust_pairplot(df, title, diag_kind='kde', sign='off'):
    # Define the color scheme
    colors = ['#f6f5f5', '#fe346e', '#512b58', '#2c003e']
    
    # Plot
    g = sns.pairplot(data=df,
                     hue='stroke', hue_order=[1, 0],
                     height=2, aspect=1,
                     corner=True, diag_kind=diag_kind,
                     palette=[colors[1], colors[2]],
                     plot_kws={'alpha': 0.85, 'size': 1, 'linewidth': 0.5, 'ec': 'black'},
                     diag_kws={'alpha': 0.95, 'ec': 'black', 'linewidth': 3})
    
    # Remove legend
    g._legend.remove()
    
    # Set facecolor
    plt.gcf().patch.set_facecolor('#f5f6f6')
    plt.gcf().patch.set_alpha(1)
    plt.gcf().set_size_inches(12, 12)
    
    # Style axes
    for ax in plt.gcf().axes:
        ax.set_facecolor('#f5f6f6')
        for loc in ['left', 'right', 'top', 'bottom']:
            ax.spines[loc].set_visible(False)
        ax.set_xlabel(xlabel=ax.get_xlabel(), **{'font': 'serif', 'size': 8, 'weight': 'bold'}, alpha=1, rotation=0)
        ax.set_ylabel(ylabel=ax.get_ylabel(), **{'font': 'serif', 'size': 8, 'weight': 'bold'}, rotation=90, alpha=1)

    # Titles and descriptions
    plt.gcf().text(0.425, 0.85, 'Healthcare Dataset: Pairwise Correlations\n{}'.format(title), {'font': 'serif', 'size': 22., 'weight': 'bold'}, alpha=1)
    plt.gcf().text(0.425, 0.77, '''The density and scatter plots provide a good visual representation\nof how certain variables (like age and glucose level)\ncorrelate more strongly with stroke risk, while other variables \nmay show weaker or No correlations.''', {'font': 'serif', 'size': 14}, alpha=1)

    # Display legend
    plt.gcf().text(0.425, 0.74, "Stroke", {'font': 'serif', 'size': 18, 'weight': 'bold', 'color': colors[1]}, alpha=1)
    plt.gcf().text(0.525, 0.74, '|', {'font': 'serif', 'size': 18, 'weight': 'bold'})
    plt.gcf().text(0.565, 0.74, "Healthy", {'font': 'serif', 'size': 18, 'weight': 'bold', 'color': colors[2]}, alpha=1)

    # Legend
    if sign == 'on':
        plt.gcf().text(0.75, -0.025, ' ', {'font': 'serif', 'size': 12, 'weight': 'bold'}, alpha=1)
    
    plt.gca().margins(x=0.25)
    plt.show()

# Box Plot
def numerical_boxplot(df, columns):
    # Check if specified columns exist in the DataFrame
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    # Check if the list of numerical columns is empty
    if not columns:
        raise ValueError("List of numerical columns is empty.")
    
    # Melt the DataFrame to long format for Seaborn boxplot
    df_melted = df.melt(id_vars='stroke', value_vars=columns, var_name='Variables', value_name='Values')
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#f5f5f5')
    
    # Plot box plots of each numerical column
    sns.boxplot(x='Variables', y='Values', hue='stroke', data=df_melted, palette=['#2c003e', '#fe346e'], width=0.4, ax=ax)
    
    # Set the title and labels
    ax.set_title('Box Plots for Numerical Variables: Stroke vs Healthy', fontsize=16, fontweight='bold', fontfamily='Serif')
    ax.set_ylabel('Values', fontsize=12, fontweight='bold', fontfamily='Serif')
    ax.set_xlabel('Variables', fontsize=12, fontweight='bold', fontfamily='Serif')
    
    # Rotate x-axis labels
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=10, fontfamily='Serif')
    ax.tick_params(axis='y', labelsize=10, labelcolor='black')
    plt.tight_layout()
    
    # Customize legend
    ax.legend(title='Stroke', title_fontsize='13', loc='upper right', prop={'family': 'Serif', 'weight': 'bold', 'size': 10})
    
    # Customizing the plot appearance
    ax.set_facecolor('#f5f5f5')
    ax.spines['bottom'].set_linewidth(2)
    for loc in ['left', 'right', 'top']:
        ax.spines[loc].set_visible(False)
        ax.spines[loc].set_linewidth(2)
    ax.tick_params(axis='x', colors='black', labelsize=10, labelrotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Healthy', 'Stroke'], title='', loc='upper right', prop={'family': 'Serif', 'weight': 'bold', 'size': 10})
    plt.show()
    
# Continous Univariate Analysis
def continuos_univariate_analysis(df, num_col):
    # Filter out continuous variables for the univariate analysis
    df_continuous = df[num_col]

    # Set up the subplot with 2 rows and 3 columns
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

    # Set background color
    fig.patch.set_facecolor('#f6f5f5')

    # Flatten the axes array
    ax = ax.flatten()

    # Loop to plot histograms for each continuous variable
    for i, col in enumerate(df_continuous.columns):
        values, bin_edges = np.histogram(df_continuous[col], 
                                        range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
        
        graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[i],
                            edgecolor='none', color='#512b58', alpha=0.8, line_kws={'lw': 3})
        
        # Add text annotations for mean and standard deviation
        ax[i].set_xlabel(col, fontsize=15, fontweight="bold", fontname="serif")
        ax[i].set_xticks(np.round(bin_edges, 1))
        ax[i].set_xticklabels(ax[i].get_xticks(), rotation=45, fontdict={'font': 'serif', 'color': 'black', 'size': 10})
        
        # Remove ylabel and yticklabels
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([])
        
        ax[i].grid(False)  # Turn off grid lines completely
        
        for j, p in enumerate(graph.patches):
            ax[i].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                        ha='center', fontsize=10, fontweight="bold", fontname="serif")
        
        textstr = '\n'.join((
            r'$\mu=%.2f$' % df_continuous[col].mean(),
            r'$\sigma=%.2f$' % df_continuous[col].std()
        ))
        ax[i].text(0.75, 0.9, textstr, transform=ax[i].transAxes, fontsize=12, verticalalignment='top',
                color='white', bbox=dict(boxstyle='round', facecolor='#512b58', edgecolor='white', pad=0.5))

        # Turn off spines for top left right
        for spine in ['top', 'right', 'left']:
            ax[i].spines[spine].set_visible(False)
            
        # Set plot background color
        ax[i].set_facecolor('#f6f5f5')
        ax[i].get_yaxis().set_visible(False)
        
    # Turn off the extra empty axes
    for i in range(len(df_continuous.columns), len(ax)):
        ax[i].axis('off')

    plt.suptitle('Distribution of Continuous Variables', fontsize=16, fontweight='bold', fontfamily='Serif',x=0.5, y=1.45)
    # Subtitle explanation
    fig.text(0.03, 1.35, '''Summary of Continuous Variables''', ha='left', fontweight='bold', fontsize=14, fontfamily='Serif')
    fig.text(-0.013, 0.95, 
            '''
            The age distribution shows that most individuals are between 49 and 65 years old, 
            with a mean of 43.23 years and a slight skew towards older ages.

            For the average glucose level, the data is heavily right-skewed, with most values 
            clustering between 55 and 100, but there is a noticeable long tail of higher glucose levels, 
            particularly over 160. The mean glucose level is 106.15, reflecting the skew in the data.

            The BMI distribution is nearly normal, centered around a mean of 28.89, indicating that most 
            individuals fall into the overweight category, with a standard deviation of 7.70. A smaller 
            group appears in the underweight or obese categories.
            ''',
            ha='left', fontsize=12, fontfamily='Serif')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Define a function to plot boxplots comparing before and after
def plot_boxplots_before_and_after(df, df_cleaned):
    # Melt the dataframes
    df_melted = df.melt(value_vars=['age', 'avg_glucose_level', 'bmi'], var_name='Variables', value_name='Values')
    df_cleaned_melted = df_cleaned.melt(value_vars=['age', 'avg_glucose_level', 'bmi'], var_name='Variables', value_name='Values')

    # Set the styling parameters
    plt.rcParams.update({
        'font.family': 'serif',
        # 'font.weight': 'bold',
        'font.size': 10,
        'figure.facecolor': '#f6f5f5'
    })

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Boxplot for df with outliers
    sns.boxplot(ax=axes[0], x='Variables', y='Values', data=df_melted, color='#512b58', width=0.4)
    axes[0].set_title('Data with Outliers', fontdict={'fontsize': 12, 'fontweight': 'bold', 'fontfamily': 'Serif'})

    # Boxplot for df_cleaned without outliers
    sns.boxplot(ax=axes[1], x='Variables', y='Values', data=df_cleaned_melted, color='#512b58', width=0.4)
    axes[1].set_title('Data without Outliers', fontdict={'fontsize': 12, 'fontweight': 'bold', 'fontfamily': 'Serif'})

    # Set the background color for each subplot
    for ax in axes:
        ax.set_facecolor('#f6f5f5')
        # ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    
# Define a function to plot continuos variable bivariate analysis
def continuos_bivariate_analysis(df, num_col):
    plt.rcParams['font.family'] = 'Serif'

    # Create the subplots
    fig, ax = plt.subplots(len(num_col), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})
    fig.set_facecolor('#f6f5f5')
    # Loop through each continuous variable to create barplots and kde plots
    for i, col in enumerate(num_col):
        # Create custom xlabel
        custom_ylabel = f'Mean {col}'
        
        # Barplot showing the mean value of the variable for each stroke category
        graph = sns.barplot(data=df, x="stroke", hue="stroke", y=col, ax=ax[i,0], palette=['#512b58', '#fe346e'], alpha=1, legend=False, width=0.7)
        
        ax[i, 0].set_xticks([0, 1])  # Set the positions of the ticks
        ax[i, 0].set_xticklabels(['Healthy', 'Stroke'], fontdict={'font': 'serif', 'color': 'black', 'size': 12})  # Set custom x-axis labels
        ax[i, 0].set_xlabel('')  # Set custom x-axis labels
        ax[i,0].set_ylabel(custom_ylabel)
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])

        # KDE plot showing the distribution of the variable for each stroke category
        sns.kdeplot(data=df[df["stroke"]==0], x=col, fill=True, linewidth=2, ax=ax[i,1], label='0', color='#512b58', alpha=1)
        sns.kdeplot(data=df[df["stroke"]==1], x=col, fill=True, linewidth=2, ax=ax[i,1], label='1', color='#fe346e', alpha=0.9)
        ax[i,1].set_yticks([])
        ax[i,1].set_ylabel('')
        ax[i,1].legend(title='Heart Disease', loc='upper right')
        
        # Set the legend
        handles, labels = ax[i,1].get_legend_handles_labels()
        ax[i,1].legend(handles, ['Healthy', 'Stroke'], title='', loc='upper right', prop={'family': 'Serif', 'weight': 'bold', 'size': 10}) 
        
        # Add mean values to the barplot
        for cont in graph.containers:
            graph.bar_label(cont, fmt='%.3g', fontsize=10)
            
        # Remove Spines
        ax[i,0].spines['top'].set_visible(False)
        ax[i,0].spines['right'].set_visible(False)
        ax[i,0].spines['left'].set_visible(False)
        ax[i,1].spines['top'].set_visible(False)
        ax[i,1].spines['right'].set_visible(False)
        ax[i,1].spines['left'].set_visible(False)
        
        # Set background color
        ax[i,0].set_facecolor('#f6f5f5')
        ax[i,1].set_facecolor('#f6f5f5')
            
    # Set the title for the entire figure
    plt.suptitle('Distribution of Continuous Variables', fontsize=16, fontweight='bold', fontfamily='Serif')
    plt.suptitle('Continuous Variables vs Target Distribution', fontsize=22, fontweight='bold', fontfamily='Serif')
    plt.tight_layout()                     
    plt.show()

# Define a function to plot stacked bar charts for categorical variables vs Stroke
def cat_bivariate_analysis(df, cat_col):
    # Sample data and colors (replace these with your actual data and color scheme)
    colors = ['#512b58', '#fe346e']
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.set_facecolor('#f6f5f5')
    # Remove 'stroke' from the categorical_features
    cat_feature = [col for col in cat_col if col != 'stroke']

    for i, col in enumerate(cat_feature):
        # Create a cross tabulation showing the proportion of purchased and non-purchased loans for each category of the feature
        cross_tab = pd.crosstab(index=df[col], columns=df['stroke'])

        # Using the normalize=True argument gives us the index-wise proportion of the data
        cross_tab_prop = pd.crosstab(index=df[col], columns=df['stroke'], normalize='index')

        # Define colormap
        cmp = sns.color_palette(colors)

        # Plot stacked bar charts
        x, y = i // 2, i % 2

        cross_tab_prop.plot(kind='bar', ax=ax[x, y], stacked=True, color=cmp, legend=False, ylabel='Proportion', sharey=True)

        # Add the proportions and counts of the individual bars to our plot
        for idx, val in enumerate(cross_tab.index.values):
            for (proportion, count, y_location) in zip(cross_tab_prop.loc[val], cross_tab.loc[val], cross_tab_prop.loc[val].cumsum()):
                if proportion > 0:  # Skip annotations with 0%
                    ax[x, y].text(
                        x=idx, 
                        y=y_location - (proportion / 2), 
                        s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                        color="lightgrey", 
                        fontsize=8, 
                        fontweight="bold", 
                        ha='center', 
                        va='center'
                    )

        ax[x, y].set_yticks([])  # Set the positions of the ticks
        ax[x, y].set_yticklabels([])  # Set custom y-axis labels
        ax[x, y].set_ylabel('')  # Set custom y-axis labels
        ax[x, y].yaxis.set_visible(False)
        ax[x, y].set_facecolor('#f5f5f5')

        # Add legend
        ax[x, y].legend(title='stroke', loc=(0.7, 0.9), fontsize=8, ncol=2)
        # Set y limit
        ax[x, y].set_ylim([0, 1.12])
        # Rotate xticks
        ax[x, y].set_xticklabels(ax[x, y].get_xticklabels(), rotation=0)
        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax[x, y].spines[spine].set_visible(False)

        # Set the legend
        handles, labels = ax[x, y].get_legend_handles_labels()
        ax[x, y].legend(handles, ['Healthy', 'Stroke'], title='', loc='upper right', prop={'family': 'Serif', 'weight': 'bold', 'size': 10}) 

    plt.suptitle('Categorical Variables vs Stroke (Stacked Barplots)', fontsize=22, fontweight='bold', fontfamily='Serif')
    plt.tight_layout()
    plt.show()

# ******************************************* Data Preprocessing Funtions ****************************************************
# Handle Outliers
def detect_and_remove_outliers(df, columns):
    total_outliers = 0
    df_cleaned = df.copy()  # Create a copy to avoid modifying the original dataframe

    print(f"Outliers detected in the following columns:")
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers for the current column
        outliers_count = ((df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)).sum()
        total_outliers += outliers_count
        
        # Remove outliers for the current column
        df_cleaned = df_cleaned[~((df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound))]
        
        # Print the column name and outliers count with 10-character formatting
        print(f"{column:<20} {outliers_count:<10}")
        
    return df_cleaned, total_outliers

# Feature Scaling
def feature_scaling(X, choice="standard", plot_scaling_effects=False):
    """
    Feature scaling using StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler.

    Parameters:
    X: Feature dataframe
    choice: 'standard', 'minmax', 'maxabs', 'robust', None
    plot_scaling_effects: True or False
    """

    if choice not in ["standard", "minmax", "maxabs", "robust", None]:
        raise ValueError(f"Unknown choice value: {choice}")

    if choice is None:
        return X

    scaler = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
    }[choice]

    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    if plot_scaling_effects:
        plt.figure(figsize=(14, 7))
        plt.gcf().set_facecolor("#f6f5f5")

        # Original Data Distributions
        ax1 = plt.subplot(1, 2, 1)
        plt.title("Original Data Distributions", fontdict={"font": "Times New Roman", "size": 16, "weight": "normal"})
        sns.histplot(data=X, color='pastel', kde=True, fill=True, alpha=0.5)
        plt.xlabel("Value", fontsize=14, fontname="Times New Roman", weight="normal")
        plt.ylabel("Density", fontsize=14, fontname="Times New Roman", weight="normal")
        ax1.set_facecolor("#f6f5f5")
        
        # Scaled Data Distributions
        ax2 = plt.subplot(1, 2, 2)
        plt.title(f"{scaler} Scaled Data Distributions", fontdict={"font": "Times New Roman", "size": 16, "weight": "normal"})
        sns.histplot(data=X_scaled_df, color='pastel', kde=True, fill=True, alpha=0.5)
        plt.xlabel("Value", fontsize=14, fontname="Times New Roman", weight="normal")
        plt.ylabel("Density", fontsize=14, fontname="Times New Roman", weight="normal")
        ax2.set_facecolor("#f6f5f5")

        # Remove the top and right spines
        sns.despine(top=True, right=True)

        plt.tight_layout()
        plt.show()

    return X_scaled_df

# Function to Handle imbalanced data, return res_df as resampled data
# Class to handle the visualisation of sampling effects
class Sampling():
    def __init__(self, feat, tar, method, ax):
        # Initialize with features, target, method name, and the axis to plot on
        self.feat = feat
        self.tar = tar
        self.method = method
        self.ax = ax

    def visualise_data(self):
        # Create a DataFrame to easily handle the target variable
        temp_y = pd.DataFrame({'y': self.tar})
        
        # Set the background color of the plot
        self.ax.set_facecolor('#f5f6f6')
                
        # Dimension reduction
        pca = PCA(n_components=2).fit_transform(self.feat)
        
        # Scatter plot for the 'Healthy' class
        self.ax.scatter(pca[temp_y['y'] == 0][:, 0], pca[temp_y['y'] == 0][:, 1], c='#512b58', s=15, linewidth=0.2, edgecolor='black')
        
        # Scatter plot for the 'Stroke' class
        self.ax.scatter(pca[temp_y['y'] == 1][:, 0], pca[temp_y['y'] == 1][:, 1], c='#fe346e', s=15, linewidth=0.2, edgecolor='black')
        
        # Remove the axis spines and ticks to clean up the plot appearance
        for loc in ['left','right','top', 'bottom']:
            self.ax.spines[loc].set_visible(False)
            self.ax.axes.get_xaxis().set_visible(False)
            self.ax.axes.get_yaxis().set_visible(False)
            self.ax.set_xticklabels('')
            self.ax.set_yticklabels('')

        # Dynamically determine the limits of the x and y axes
        self.ax.set_xlim(xmin = -6, xmax = 6)
        self.ax.set_ylim(ymin = -5, ymax = 6)

        # Add text labels for the 'Stroke' and 'Healthy' categories
        self.ax.text(1.6,5.5,"Stroke", fontweight="bold", fontfamily='serif', fontsize=13, color='#ff005c')
        self.ax.text(3.2,5.5,"|", fontweight="bold", fontfamily='serif', fontsize=13, color='black')
        self.ax.text(3.4,5.5,"Healthy", fontweight="bold", fontfamily='serif', fontsize=13, color='#512b58')

        # Add title and dataset description with increased spacing
        self.ax.text(-6,7.2,self.method, {'font': 'serif', 'weight': 'bold', 'size': 20}, alpha = 0.8)
        self.ax.text(-6,6.2,'{} contain {} number of datapoint, \nand targets distribution as {}.'.format(self.method,len(self.feat), {0:Counter(self.tar)[0], 1:Counter(self.tar)[1]}), {'font': 'serif', 'weight': 'normal', 'size': 12}, alpha = 0.7)
        return self.ax

def SMOTE_resample(df, target_column_name='stroke', neighbors=5):
    # Separate the features and the target variable from the dataset
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Initialize SMOTE with desired settings
    sm = SMOTE(sampling_strategy='auto', k_neighbors=neighbors, random_state=101)
    
    # Apply SMOTE to generate synthetic samples
    X_res, y_res = sm.fit_resample(X, y)

    # Combine the resampled features and target back into a DataFrame
    res_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_column_name)], axis=1)

    return res_df

# ******************************************* Feature Selection Funtions ****************************************************
def visualise_selected_features(feature_names, scores, top_indices):
    """
    Function to visualise the selected variables based on their importance scores.
    Parameters:
        variable_names (list): List of variable names.
        scores (array): Array of variable importance scores.
        top_indices (list): List of indices of the top selected variables.
    
    Returns:
        str: A base64-encoded string of the plot image representing the selected variables.
    """
    top_scores = scores[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    plt.gcf().set_facecolor('#f6f5f5')  # Correct way to set the face color

    plt.figure(figsize=(12, 6))
    plt.barh(top_features, top_scores, color='skyblue')
    plt.xlabel('Feature Importance Score', fontdict={'font': 'Times New Roman', 'size': 12, 'weight': 'normal'})
    plt.title('Top Selected Variables', fontdict={'font': 'Times New Roman', 'size': 16, 'weight': 'bold'})
    plt.show()

# Train Test Split
def split_data(df, target_column, cat_list, num_list, test_size=0.2, k_best=None):
    '''
    Function to split the data into train and test sets, apply feature scaling, and feature selection.
    
    Args:
    df (pandas.DataFrame): The input dataset
    target_column (str): The name of the target column
    test_size (float): The proportion of the dataset to include in the test split (default: 0.2)
    scaling_choice (str): The type of feature scaling to apply (default: 'standard')
    plot_scaling_effects (bool): Whether to plot the distribution of the original and scaled data (default: False)
    k_best (int): Number of top features to select (default: None, which means no feature selection)
    data_analysis (bool): Whether to perform data analysis to get column types (default: False)
    experiment_name (str): The name of the experiment (default: '')
    
    Returns:
    X_train (pandas.DataFrame): The training features
    X_test (pandas.DataFrame): The test features
    y_train (pandas.Series): The training target
    y_test (pandas.Series): The test target
    '''

    # Ensure target column is not in categorical list
    if target_column in cat_list:
        cat_list.remove(target_column)
    
    # Separate the features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    top_features = X.columns  

    if k_best is not None:
        # Calculate the total number of features
        total_features = len(num_list) + len(cat_list)
        
        # Validate k_best
        if k_best <= 0 or k_best > total_features:
            k_best = total_features
            
        # Feature selection for continuous features
        if num_list:
            X_continuous = X[num_list]
            selector_continuous = SelectKBest(score_func=f_classif, k='all')
            selector_continuous.fit_transform(X_continuous, y)
            scores_continuous = selector_continuous.scores_
        else:
            # X_continuous_selected = np.array([])
            scores_continuous = np.array([])
        
        # Feature selection for categorical features
        if cat_list:
            X_categorical = X[cat_list]
            selector_categorical = SelectKBest(score_func=chi2, k='all')
            selector_categorical.fit_transform(X_categorical, y)
            scores_categorical = selector_categorical.scores_
        else:
            scores_categorical = np.array([])
        
        # Combine scores and select top k features
        combined_scores = np.concatenate((scores_continuous, scores_categorical))
        feature_names = num_list + cat_list
        
        top_indices = np.argsort(combined_scores)[-k_best:]
        top_features = [feature_names[i] for i in top_indices]
        
        # Visualise the selected features
        visualise_selected_features(feature_names, combined_scores, top_indices)
        
        X = X[top_features]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
    
    return X_train, X_test, y_train, y_test

# ******************************************* Feature Importance Analysis ****************************************************   
import matplotlib.lines as lines
def feature_importance_analysis(feature_importance_df, insight):
    # Define colors
    colors = ['grey', '#512b58', '#fe346e']
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    
    # Import the lines module to create a vertical line on the right side of the figure

    # Set background color for the figure
    background_color = "#f5f5f5"

    # Create a figure and axis with the specified size and background color
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=background_color)

    # Get the number of unique features in the dataset
    num_features = feature_importance_df['Feature'].nunique()

    # Generate a list of colors from the colormap in reverse order
    palette = [colormap(i/num_features) for i in reversed(range(num_features))]

    # Create a barplot showing the feature importance, using the custom color palette
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax, palette=palette, hue='Feature', legend=False, width=0.6)

    # Set the background color of the axis
    ax.set_facecolor(background_color)

    # Hide the top, left, and right spines of the plot (borders)
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Add a bold, serif font title at the top left of the figure
    fig.text(0.12, 0.92, "Feature Importance: LightGBM Stroke Prediction", fontsize=18, fontweight='bold', fontfamily='serif')

    # Set empty x and y labels with light font style, removing the x-axis label's default positioning
    plt.xlabel(" ", fontsize=12, fontweight='light', fontfamily='serif', loc='left', y=-1.5)
    plt.ylabel(" ", fontsize=12, fontweight='light', fontfamily='serif')

    # Add a bold, serif font subtitle at the top right of the figure (aligned with text on the right)
    fig.text(1.1, 0.92, 'Insight', fontsize=18, fontweight='bold', fontfamily='serif')

    # Add a multiline text block explaining the insights about the feature importance
    fig.text(1.1, 0.315, insight, fontsize=14, fontweight='light', fontfamily='serif')

    # Remove tick marks from both axes
    ax.tick_params(axis='both', which='both', length=0)
    # Add a thin black line on the right side of the figure to separate the insight text from the plot
    l1 = lines.Line2D([0.98, 0.98], [0, 1], transform=fig.transFigure, figure=fig, color='black', lw=0.2)
    fig.lines.extend([l1])

    # Display the final plot with the adjustments
    plt.show()