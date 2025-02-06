# **Data Visualization Dashboard**

This dashboard offers a flexible solution for visualizing various data types from a given dataset. It supports different combinations of numerical and categorical variables, enabling comprehensive insights through a wide range of visualizations.

---

## **Supported Variable Combinations**

### **1. Numeric Variables**
- **Single Numeric Variable:** Histogram, density plot.
- **Two Numeric Variables:**  
  - Ordered: Line plot.  
  - Unordered (with less than 2000 data points): Boxplot, histogram, scatter plot.  
  - Unordered (with more than or equal 2000 data points): Violin plot, density plot, scatter with marginal points, 2D density plot.
- **Three Numeric Variables:** 
  - Unordered: Boxplot, violin plot, bubble plot, correlogram, 3D scatter plot.
  - Ordered: Line plot.
- **Several Numeric Variables:**  
  - Ordered: Line plot.  
  - Unordered: Boxplot, violin plot, ridgeline plot, correlation heatmap, dendrogram, dendrogram with heatmap, PCA scatter plot.

Note: Ordered means that at least one of the selected variables is ordered. That variable is used as the x axis variable for a line graph.
### **2. Categorical Variables**
- **One Categorical Variable:** Bar plot, lollipop chart, pie chart, donut chart. Pie and Donut chart are made only if the selected variable has up to 10 unique categorical variables.
- **Two Categorical Variables:** Grouped bar plot, Grouped stacked bar plot.

### **3. Numeric & Categorical Combinations**
- **One Numeric + One Categorical:** Boxplot, violin plot, ridgeline plot, density plot, histogram.
- **One Categorical + Several Numeric Variables:** Grouped box plots, grouped violin plots, PCA scatter plot (or regular scatter plot for 2 numerical variables), grouped correlogram.
- **One Numeric + Several Categorical Variables (up to 3 categorical variables):** Grouped box plots, split violin plot

Note: A combination of two or more numeric and two or more categoric variables is not supported yet.

---

## **Getting Started**

### **1. Requirements Installation**
- Install the needed requirements from the file requirements.txt by running: pip install -r requirements.txt

### **2. Data Preparation**
- Ensure your data is ready and accessible in a suitable format (e.g., CSV).
- Copy the file path of the dataset and paste it into the `DATA_PATH` variable in `Dashboard.py`.

## **Usage Instructions**
1. Run the `Dashboard.py` file in your Python environment.
2. Click the generated dashboard link in the code output to access the interface.
3. Select the desired imputation method.
4. Choose the variables to visualize.
5. Click the "Generate Visualizations" button to display the plots.

---
### **Imputation Methods**
Select one of the following methods for handling missing values:
- **Drop Missing Values:** Removes any rows with missing data.
- **Simple Imputer (Mean):** Replaces missing values with the column mean.
- **KNN Imputer:** Uses the K-nearest neighbors algorithm for imputation.
- **Iterative Imputer:** Uses multiple regression models to estimate missing values.

---

## **Notes**
- **Some data types aren't supported yet.** The variables you choose must be either numerical (data type: 'int64', 'float64', 'int32', 'float32', 'int16', 'float16') or categorical (data type: object, category or bool). Otherwise, they will be considered unclassified and will result in no visualisations being generated or errors. 

Enjoy visualizing your data!
