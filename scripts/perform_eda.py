#%%[markdown]
# Perform Exploratory Data Analysis (EDA)
# %%[markdown]
## On the Raw Data
# %%
# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Load the preprocessed data
df=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/raw_data.csv')
# %%
#percentages of different categories
male_percentage=df[df['Gender']=='M'].shape[0]/df.shape[0]*100
female_percentage=df[df['Gender']=='F'].shape[0]/df.shape[0]*100
survived_percentage=df[df['Hospital_Expire_Flag']==0].shape[0]/df.shape[0]*100
died_percentage=df[df['Hospital_Expire_Flag']==1].shape[0]/df.shape[0]*100
emergency_percentage=df[df['Admission_Type']=='EMERGENCY'].shape[0]/df.shape[0]*100
urgent_percentage=df[df['Admission_Type']=='URGENT'].shape[0]/df.shape[0]*100
elective_percentage=df[df['Admission_Type']=='ELECTIVE'].shape[0]/df.shape[0]*100

# %%
# Set global text sizes
plt.rcParams['font.size'] = 12  # Sets the default font size
plt.rcParams['axes.labelsize'] = 24  # Sets the font size for axes labels
plt.rcParams['axes.titlesize'] = 24  # Sets the font size for the title
plt.rcParams['xtick.labelsize'] = 24  # Sets the font size for the x-tick labels
plt.rcParams['ytick.labelsize'] = 24  # Sets the font size for the y-tick labels
# %%
#Gender distribution
df['Gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xticks(ticks=[0,1],labels=['Male','Female'],rotation=0)
plt.ylabel('Count')
plt.bar_label(container=plt.gca().containers[0],label_type='edge',labels=[f'{male_percentage:.1f}%',f'{female_percentage:.1f}%'])
plt.savefig('plot1.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
#%%
#Age distribution
df['Age_at_Admission'].plot(kind='hist',bins=40)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('plot2.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
#%%
#Admission Type distribution
df['Admission_Type'].value_counts().plot(kind='bar',rot=0)
plt.title('Admission Type Distribution')
plt.ylabel('Count')
plt.bar_label(container=plt.gca().containers[0],label_type='edge',labels=[f'{emergency_percentage:.1f}%',f'{elective_percentage:.1f}%',f'{urgent_percentage:.1f}%'])
plt.savefig('plot3.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
#Survived or died distribution
df['Hospital_Expire_Flag'].value_counts().plot(kind='bar')
plt.title('Survived or Died Distribution')
plt.xticks(ticks=[0,1],labels=['Survived','Died'],rotation=0)
plt.ylabel('Count')
plt.bar_label(container=plt.gca().containers[0],label_type='edge',labels=[f'{survived_percentage:.1f}%',f'{died_percentage:.1f}%'])
plt.savefig('plot4.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
#%%
#ICU Length of stay distribution
df['ICU_Length_of_Stay'].plot(kind='hist',bins=40)
plt.title('ICU Length of Stay Distribution')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Count')
plt.savefig('plot5.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
#Finding outliers
sns.boxplot(x=df['Age_at_Admission'])
plt.savefig('plot6.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['ICU_Length_of_Stay'])
plt.savefig('plot7.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Heart_Rate'])
plt.savefig('plot8.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Blood_Pressure'])
plt.savefig('plot9.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Hemoglobin'])
plt.savefig('plot10.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Sodium'])
plt.savefig('plot11.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Potassium'])
plt.savefig('plot12.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
sns.boxplot(x=df['Avg_Med_Dose'])
plt.savefig('plot13.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
#%%
sns.boxplot(x=df['Num_Diagnoses'])
plt.savefig('plot14.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
#%%
sns.boxplot(x=df['Num_Procedures'])
plt.savefig('plot15.png',dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure window to free up memory
# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of your saved plot filenames
plot_files = ['plot1.png', 'plot2.png', 'plot3.png', 'plot4.png', 
              'plot5.png', 'plot6.png', 'plot7.png', 'plot8.png',
              'plot9.png', 'plot10.png', 'plot11.png', 'plot12.png',
              'plot13.png', 'plot14.png']

# Number of rows and columns
nrows = 5
ncols = 3

# Set up the subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25))  # Adjust figsize for an optimal display

# Flatten the axes array and iterate over it, along with the filenames
for ax, plot_file in zip(axes.flatten(), plot_files + [None] * (nrows*ncols - len(plot_files))):  # Padding the list to match the grid size
    if plot_file:
        # Read the image file
        img = mpimg.imread(plot_file)
        # Display the image
        ax.imshow(img)
    # Remove axes for a cleaner look
    ax.axis('off')

# Adjust layout with less padding
plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between plots
plt.show()


# %%[markdown]
## On the Preprocessed Data
# %%
# Load the preprocessed data
df=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/preprocessed_data.csv')
# %%

