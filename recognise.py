# Function to classify columns
def classify_columns(df):
    column_types = {
        "categorical": [],
        "numerical_continuous": [],
        "numerical_discrete": [],
        "numerical_categorical": [],  # Added for variables with <= 15 unique values
        "unclassified": []  # Placeholder for columns that don't fit any of the above
    }
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object' or dtype == 'category':
            column_types['categorical'].append(col)
        elif dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'int8']:
            # Check unique values ratio for continuous vs discrete
            unique_values = df[col].nunique()
            total_values = df[col].notna().sum()  # Count non-missing values
            unique_ratio = unique_values / total_values
            
            if unique_ratio > 0.1:  # Arbitrary threshold for continuous
                column_types['numerical_continuous'].append(col)
             # If the unique values are <= 15, classify as numerical categorical
            elif unique_values <= 15:
                column_types['numerical_categorical'].append(col)
            else:
                column_types['numerical_discrete'].append(col)
        elif dtype == 'bool':
            column_types['categorical'].append(col)
        else:
            column_types['unclassified'].append(col)
        

    return column_types

# numeric

plot_options = {
    # numerical
    "one_numeric": ["histogram", "density_plot"],
    "two_numeric_ordered": ["connected_scatter_plot", "area_plot", "line_plot"],
    "two_numeric_not_ordered_few_points": ["boxplot", "histogram", "scatter_plot"],
    "two_numeric_not_ordered_many_points": ["violin_plot", "density_plot", "scatter_with_marginal_points", "2d_density_plot"],

    "three_numeric_ordered": ["stream_graph", "stacked_area_plot", "area_plot", "line_plot"],
    "three_numeric_not_ordered": ["boxplot", "violin_plot", "bubble_plot","correlogram", "3d_scatter_plot"],
    "several_numeric_ordered": ["stream_graph", "stacked_area_plot", "area_plot", "line_plot"], # only line plot implemented
    "several_numeric_not_ordered": ["boxplot", "violin_plot", "ridgeline", "correlogram", "corr_heatmap", "dendrogram", "PCA_scatter_plot"],

    # categorical only
    "one_categorical": ["barplot", "lollipop", "waffle", "wordcloud", "doughnut", "pie", "treemap", "circular_packing"],
    "several_categorical_independent_lists": ["venn_diagram"], # not implemented in the dashoard
    "several_categorical_hierarchical": ["treemap", "circular_packing", "sunburst", "barplot", "dendrogram"], # not implemented in the dashoard
    "several_categorical_adjacency": ["network", "chord", "arc", "sankey", "heatmap"], # not implemented in the dashoard
    "several_categorical_subgroup": ["grouped_barplot", "stacked_barplot"],

    #numerical and categorical
    "one_numeric_one_categorical_single": ["boxplot", "lollipop"], 
    "one_numeric_one_categorical_multiple": ["boxplot", "violin_plot", "ridgeline", "density_plot", "histogram"], 
    
    "one_categorical_several_numeric_single": ["grouped_scatter_plot", "heatmap", "lollipop", "grouped_barplot", "stacked_barplot", "parallel_plot", "spider_plot", "sankey"], # only grouped scatter plot implemented 
    "one_categorical_several_numeric_unordered": ["boxplot", "violin_plot","grouped_scatter_plot","PCA_scatter_plot", "correlogram"],
    "one_categorical_several_numeric_ordered":["stacked_area_plot", "area_plot", "stream_graph", "line_plot", "connected_scatter_plot"], # only line plot implemented

    "one_numeric_several_categorical_subgroup_single": ["grouped_scatter_plot", "heatmap", "lollipop", "grouped_barplot", "stacked_barplot", "parallel_plot", "spider_plot", "sankey"], # not implemented yet
    "one_numeric_several_categorical_subgroup_multiple": ["grouped_boxplot", "grouped_violin_plot"],
    "one_numeric_several_categorical_hierarchical_single": ["barplot", "dendrogram","sunburst", "treemap", "circular_packing"], # not implemented
    "one_numeric_several_categorical_hierarchical_multiple": ["grouped_boxplot", "grouped_violin_plot"],
    "one_numeric_several_categorical_adjacency": ["network", "chord", "arc", "sankey", "heatmap"] # not implemented yet
}


# Function to check if numeric columns are ordered
def is_any_column_ordered(df, columns):
    return any(df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing for col in columns)


# Function to check observations per group
def check_observations_per_group(df, categorical_col, threshold=0.95):
    unique_groups = df[categorical_col].nunique()
    total_rows = len(df)
    unique_proportion = unique_groups / total_rows
    if unique_proportion >= threshold:
        return "single"
    else:
        return "multiple"
def check_obs_helper(df, categorical_cols, threshold=0.95):
    # used for checking if any of the categorical columns have multiple observations per group
    result = {}
    for col in categorical_cols:
        result[col] = check_observations_per_group(df, col, threshold)
    # Check if any categorical column has multiple observations per group
    multiple_obs_columns = {col: res for col, res in result.items() if res == 'multiple'}
    return multiple_obs_columns

# Function to check if a column is a subgroup of another column
def is_subgroup(df, col1, col2, threshold=0.5):
    """
    Check if one of the two categorical columns is a subgroup of the other.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        threshold (float): Threshold for significant reduction in unique values.
                          Default is 0.5 (50%).

    Returns:
        bool: True if one column is a subgroup of the other, False otherwise."""
    
    unique_vals_col1 = df[col1].nunique()
    unique_vals_col2 = df[col2].nunique()

    if unique_vals_col1 >= unique_vals_col2:
        child_col = col1
        parent_col = col2
    else:
        parent_col = col1
        child_col = col2
    
    unique_parent_vals = df[parent_col].nunique()
    unique_child_vals = df[child_col].nunique()
    
    if unique_child_vals >= threshold * unique_parent_vals:
        return True
    else:
        return False
    
    
def subgroup_helper(df, categorical_cols):
    checked_combinations = set()
    subgroups = []

    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if i != j and (col1, col2) not in checked_combinations and (col2, col1) not in checked_combinations:
                checked_combinations.add((col1, col2))
                if is_subgroup(df, col1, col2):
                    subgroups.append((col1, col2))   
    return subgroups


########################################## main function for numerical only ########################################################
def find_numerical(df):
    column_types = classify_columns(df)
    # Filter for numerical columns
    numerical_cols = column_types['numerical_continuous'] + column_types['numerical_discrete'] # Changed to also include discrete variables that have more than 15 unique values and less than 10% unique value ratio

    # Function to check if any variable has all non-negative values -> important for bubble charts because the size can't be negative!
    def has_non_negative_variable(columns):
        return any((df[col] >= 0).all() for col in columns)
    # Case: Only one numeric variable
    if len(numerical_cols) == 1:
        return {"columns": numerical_cols, "plots": plot_options["one_numeric"]}
    # Case: Two numeric variables
    if len(numerical_cols) == 2:
        ordered = is_any_column_ordered(df, numerical_cols)
        # If ordered, suggest appropriate plots
        if ordered:
            return {"columns": numerical_cols, "plots": plot_options["two_numeric_ordered"]}
        # If not ordered, check the number of data points
        total_data_points = df[numerical_cols].notna().sum().min()  # Handle missing values
        if total_data_points < 2000:
            return {"columns": numerical_cols, "plots": plot_options["two_numeric_not_ordered_few_points"]}
        else:
            return {"columns": numerical_cols, "plots": plot_options["two_numeric_not_ordered_many_points"]}
    # Case: Three numeric variables
    if len(numerical_cols) == 3:
        ordered = is_any_column_ordered(df, numerical_cols)
        available_plots = plot_options["three_numeric_ordered"] if ordered else plot_options["three_numeric_not_ordered"]
        # Remove "bubble_plot" if no variable has all non-negative values
        if not has_non_negative_variable(numerical_cols) and "bubble_plot" in available_plots:
            available_plots = [plot for plot in available_plots if plot != "bubble_plot"]
        return {"columns": numerical_cols, "plots": available_plots}
    # Case: More than three numeric variables (several variables)
    if len(numerical_cols) > 3:
        ordered = is_any_column_ordered(df, numerical_cols)
        # If ordered, suggest appropriate plots
        if ordered:
            return {"columns": numerical_cols, "plots": plot_options["several_numeric_ordered"]}
        else:
            return {"columns": numerical_cols, "plots": plot_options["several_numeric_not_ordered"]}

        
########################################## main function for categorical only ########################################################
def find_categorical(df):
    column_types = classify_columns(df)
    categorical_cols = column_types["categorical"] + column_types["numerical_categorical"]
    # Case: One categorical variable
    if len(categorical_cols) == 1:
        return {"columns": categorical_cols, "plots": plot_options["one_categorical"]}
    # Case: Several independent categorical variables
    if len(categorical_cols) > 1:
        subgroups = subgroup_helper(df, categorical_cols)
        if len(subgroups) > 0:
            print("Subgroups found: ", subgroups)
            return {"columns": categorical_cols, "plots": plot_options["several_categorical_subgroup"]}
        else:
            print("No subgroups found.")
            return {"columns": categorical_cols, "plots": plot_options["several_categorical_independent_lists"]} # adjacency and hierarchy still aren't implemented

###################################### main function for numeric and categorical #############################################

def find_numeric_and_categorical(df):
    column_types = classify_columns(df)
    numeric_cols = column_types["numerical_continuous"] + column_types["numerical_discrete"]
    categorical_cols = column_types["categorical"] + column_types["numerical_categorical"]
    # Case: One numeric and one categorical variable
    if len(numeric_cols) == 1 and len(categorical_cols) == 1:
        numeric_col = numeric_cols[0]
        categorical_col = categorical_cols[0]
        obs_type = check_observations_per_group(df, categorical_col)
        plot_key = "one_numeric_one_categorical_" + obs_type
        return {"columns": {"numeric": numeric_col, "categorical": categorical_col}, "plots": plot_options[plot_key]}
    # Case: One categorical and several numeric variables
    if len(numeric_cols) > 1 and len(categorical_cols) == 1:
        categorical_col = categorical_cols[0]
        obs_type = check_observations_per_group(df, categorical_col)
        if obs_type == "single":
            return {"columns": {"numeric": numeric_cols, "categorical": categorical_col}, "plots": plot_options["one_categorical_several_numeric_single"]}
        else:
            ordered = is_any_column_ordered(df, numeric_cols)
            if ordered:
                return {"columns": {"numeric": numeric_cols, "categorical": categorical_col}, "plots": plot_options["one_categorical_several_numeric_ordered"]}
            else:
                return {"columns": {"numeric": numeric_cols, "categorical": categorical_col}, "plots": plot_options["one_categorical_several_numeric_unordered"]}
    # Case: One numeric and several categorical variables
    if len(numeric_cols) == 1 and len(categorical_cols) > 1:
        numeric_col = numeric_cols[0]
        """multiple_obs_columns = check_obs_helper(df, categorical_cols)
        if multiple_obs_columns:
            return {"columns": {"numeric": numeric_col, "categorical": categorical_cols}, "plots": plot_options["one_numeric_several_categorical_subgroup_multiple"]}"""
        subgroups = subgroup_helper(df, categorical_cols)
        if subgroups:
            multiple_obs_columns = check_obs_helper(df, categorical_cols)
            if multiple_obs_columns:
                return {"columns": {"numeric": numeric_col, "categorical": categorical_cols}, "plots": plot_options["one_numeric_several_categorical_subgroup_multiple"]}
            else:
                return {"columns": {"numeric": numeric_col, "categorical": categorical_cols}, "plots": plot_options["one_numeric_several_categorical_subgroup_single"]}
        else:
            return {"columns": {"numeric": numeric_col, "categorical": categorical_cols}, "plots": plot_options["one_numeric_several_categorical_adjacency"]} # not implemented in the dashboard
        # hierarchichal still not implemented

        
    
       
