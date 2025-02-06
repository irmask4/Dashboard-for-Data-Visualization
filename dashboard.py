import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from recognise import classify_columns, find_numerical, find_categorical, find_numeric_and_categorical
from plots import create_2d_density_plot, create_ridgeline_plot, create_line_plot_multiple_vars, create_line_plot, create_pairplot, create_dendrogram_with_heatmap, create_PCA_scatter_plot, create_lollipop_plot, create_sorted_bar_plot, create_grouped_bar_plot, create_grouped_stacked_bar_plot, create_boxplot, create_violin_plot, create_kde_plot, create_histogram_plot,create_grouped_boxplot, create_grouped_violinplot

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer



# Load the dataset
DATA_PATH = 'Datasets/Data_Cortex_Nuclear.csv'
#DATA_PATH = 'Datasets/Heart_failure_clinical_records_dataset.csv' # allows testing line graphs for ordered variables ('time')
#DATA_PATH = 'Datasets/diabetic_data.csv'
df = pd.read_csv(DATA_PATH)
###########################################################################################################
# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Data Visualization Dashboard"

app.layout = html.Div([
    html.H1("Data Visualization Dashboard", style={'textAlign': 'center'}),

    # Radio buttons for imputation methods
    html.Label("Select Imputation Method:"),
    dcc.RadioItems(
        id="imputation-method",
        options=[
            {"label": "Drop Missing Values", "value": "drop"},
            {"label": "Simple Imputer (Mean)", "value": "simple"},
            {"label": "KNN Imputer", "value": "knn"},
            {"label": "Iterative Imputer", "value": "iterative"}
        ],
        value="drop",
        inline=True
    ),
    
    # Button to Select All Variables
    html.Button("Select All", id="select-all-btn", n_clicks=0, style={'position': 'absolute',
                    'right': '20px',  # Adjusts the button's distance from the right edge
                    }),

    # Dropdown to select variables
    html.Label("Select Variables for Visualization:"),
    dcc.Dropdown(
        id="variable-dropdown",
        options=[{"label": col, "value": col} for col in df.columns], 
        multi=True,
        placeholder="Select one or more variables..."
    ),

    # Button to generate visualizations
    html.Button("Generate Visualizations", id="generate-btn", n_clicks=0),
    

    # Div to display the selected plots
    html.Div(id="visualization-output")
])
# callback for selecting all variables
@app.callback(
    Output("variable-dropdown", "value"),  # Updates the dropdown's selected values
    Input("select-all-btn", "n_clicks")    # Triggered by the "Select All Variables" button
)
def handle_select_all(n_clicks):
    if n_clicks > 0:  # Check if the button has been clicked
        return df.columns.tolist()  # Select all variables
    return dash.no_update  # Do nothing if button is not clicked

# Callback for handling imputation and updating the dataset
@app.callback(
    Output("visualization-output", "children"),
    Input("generate-btn", "n_clicks"),
    State("imputation-method", "value"),
    State("variable-dropdown", "value")
)
def update_visualizations(n_clicks, imputation_method, selected_vars):
    if not selected_vars:
        return html.Div("Please select variables to visualize.", style={"color": "red"})
    # Handle missing data based on selected imputation method
    df_selected = df[selected_vars]
    numerical_cols = df_selected.select_dtypes(include=["number"]).columns
    non_numerical_cols = df_selected.select_dtypes(exclude=["number"]).columns

    # Handle missing data for numerical columns
    if imputation_method == "drop":
        selected_df = df_selected.dropna()
    elif imputation_method == "simple":
        imputer = SimpleImputer(strategy="mean")
        numerical_data = pd.DataFrame(imputer.fit_transform(df_selected[numerical_cols]), columns=numerical_cols)
        selected_df = pd.concat([numerical_data, df_selected[non_numerical_cols]], axis=1)
    elif imputation_method == "knn":
        imputer = KNNImputer(n_neighbors=5)
        numerical_data = pd.DataFrame(imputer.fit_transform(df_selected[numerical_cols]), columns=numerical_cols)
        selected_df = pd.concat([numerical_data, df_selected[non_numerical_cols]], axis=1)
    elif imputation_method == "iterative":
        imputer = IterativeImputer()
        numerical_data = pd.DataFrame(imputer.fit_transform(df_selected[numerical_cols]), columns=numerical_cols)
        selected_df = pd.concat([numerical_data, df_selected[non_numerical_cols]], axis=1)
    else:
        return html.Div("Invalid imputation method selected.", style={"color": "red"})

    # Classify columns using the recognise.py logic
    column_types = classify_columns(selected_df)
    numerical = column_types['numerical_continuous'] + column_types['numerical_discrete']
    categorical = column_types['categorical'] + column_types['numerical_categorical']

    if numerical and not categorical: 
        numerical_plot_suggestions = find_numerical(selected_df)
        plots = numerical_plot_suggestions.get("plots", [])
    elif categorical and not numerical:
        categorical_plot_suggestions = find_categorical(selected_df)
        plots = categorical_plot_suggestions.get("plots", [])
    if numerical and categorical:
        other_keys_empty = all(not column_types[key] for key in column_types if key not in ['numerical_continuous', 'numerical_discrete','categorical', 'numerical_categorical']) # Ensure other keys are empty
        if other_keys_empty:
            mixed_plot_suggestions = find_numeric_and_categorical(selected_df)
            plots = mixed_plot_suggestions.get("plots", [])
        else:
            print("Unclassified variable is present.")

      ################################################################################################################################
    # Generate visualizations based on the suggested plots
    visualizations = []
    for plot in plots:
        ######################## one or two numeric variables
        if plot == "histogram":
            create_histogram_plot(visualizations, selected_df, numerical, categorical)
        elif plot == "density_plot":
            create_kde_plot(visualizations, selected_df, numerical, categorical)
        ################################################################# two numeric variables        
        elif plot == "scatter_plot" and len(selected_vars) == 2:
            fig = px.scatter(selected_df, x=selected_vars[0], y=selected_vars[1], 
                             title=f"Scatter Plot of {selected_vars[0]} vs {selected_vars[1]}")
            fig.update_traces(marker=dict(opacity=0.7))
            fig.update_layout(template="plotly_white")
            visualizations.append(dcc.Graph(figure=fig))
        elif plot == "boxplot":
            create_boxplot(visualizations, selected_df, numerical, categorical)
        elif plot == "violin_plot":   
            create_violin_plot(visualizations, selected_df, numerical, categorical)
        elif plot == "scatter_with_marginal_points" and len(selected_vars) == 2:
            fig = px.scatter(selected_df, x=selected_vars[0], y=selected_vars[1], marginal_x="histogram", marginal_y="histogram", color_discrete_sequence=["#4aa3d9"], 
                             title=f"Scatter Plot with Marginal Points of {selected_vars[0]} vs {selected_vars[1]}")
            visualizations.append(dcc.Graph(figure=fig))
        elif plot == "2d_density_plot" and len(selected_vars) == 2:
            x_col, y_col = selected_vars
            fig = create_2d_density_plot(selected_df, x_col, y_col, grid_size=100)
            visualizations.append(dcc.Graph(figure=fig))
        ################################################################# three numeric variables
        elif plot == "bubble_plot" and len(selected_vars) == 3:
            non_negative_var = None
            for var in selected_vars:
                if (selected_df[var] >= 0).all():  # Check if the column has all non-negative values
                    non_negative_var = var
                    break  # Once a non-negative variable is found, break the loop
            if non_negative_var:
                # Determine which variables to use for x and y axes
                x_var, y_var = [var for var in selected_vars if var != non_negative_var]
                # Create the bubble plot using the non-negative variable for size and color
                fig = px.scatter(selected_df, x=x_var, y=y_var,
                                size=selected_df[non_negative_var], size_max=20, opacity=0.6,
                                color=selected_df[non_negative_var], color_continuous_scale="viridis",
                                title=f"Bubble Plot of {x_var} vs {y_var} (Size and Color: {non_negative_var})")
                visualizations.append(dcc.Graph(figure=fig))

        elif plot == "3d_scatter_plot" and len(selected_vars) == 3:
            fig = px.scatter_3d(selected_df, x=selected_vars[0], y=selected_vars[1], z=selected_vars[2], color=selected_vars[2], color_continuous_scale="viridis", width=900, height=900,
                                title=f"3D Surface Plot of {selected_vars[0]} vs {selected_vars[1]} vs {selected_vars[2]}")
            visualizations.append(dcc.Graph(figure=fig))
        ################################################################# several numeric variables
        elif plot == "ridgeline":
            create_ridgeline_plot(visualizations, selected_df, numerical, categorical)
        elif plot == "correlogram": 
            if len(categorical) == 0 and len(numerical) <= 8:
                create_pairplot(visualizations, selected_df, numerical, categorical)
            elif len(categorical) == 1 and len(numerical) <= 8:
                create_pairplot(visualizations, selected_df, numerical, categorical)
        elif plot == "corr_heatmap":
            fig = px.imshow(selected_df.corr(), color_continuous_scale="jet", title="Correlation Heatmap", text_auto=".2f")
            fig.update_layout(height=650, width=650)
            visualizations.append(dcc.Graph(figure=fig))
        elif plot == "dendrogram":
            corr = selected_df.corr()
            corr_array = np.array(corr)
            d = ff.create_dendrogram(corr_array, orientation='left', labels=corr.columns)
            d.update_layout(width=650, height=650)
            visualizations.append(dcc.Graph(figure=d))
       ###################### add clustered correlation heatmap
        elif plot == "PCA_scatter_plot":    
            create_PCA_scatter_plot(visualizations, selected_df, numerical, categorical)    
        ###########################################line plots
        elif plot == "line_plot" and len(selected_vars) == 2:
            try:
                # Generate the line plot using the helper function
                fig = create_line_plot(selected_df, selected_vars)
                visualizations.append(dcc.Graph(figure=fig))
            except ValueError as e:
                # Handle the case where no ordered variable is found
                visualizations.append(html.Div(f"Error: {str(e)}"))
        elif plot == "line_plot" and len(selected_vars) > 2:
            fig = create_line_plot_multiple_vars(selected_df, selected_vars)
            visualizations.append(dcc.Graph(figure=fig))
        ################################################################################################################################ categoric only
        elif plot == "barplot":
            if len(selected_vars) == 1:
                fig = create_sorted_bar_plot(selected_df, selected_vars[0])
                visualizations.append(dcc.Graph(figure=fig))
        elif plot == "lollipop":
            create_lollipop_plot(visualizations, selected_df, numerical, categorical)
        elif plot == "doughnut" and len(selected_vars) == 1:
            if selected_df[selected_vars[0]].nunique() <= 10:
                fig = px.pie(selected_df, names=selected_vars[0], hole=0.5, title=f"Doughnut Chart of {selected_vars[0]}")
                visualizations.append(dcc.Graph(figure=fig))
        elif plot == "pie" and len(selected_vars) == 1:
            if selected_df[selected_vars[0]].nunique() <= 10:
                fig = px.pie(selected_df, names=selected_vars[0], title=f"Pie Chart of {selected_vars[0]}")
                visualizations.append(dcc.Graph(figure=fig))

        ################################################################################################################################# cat and numeric
        elif plot == "grouped_scatter_plot":
            if len(numerical) == 2 and len(categorical) == 1:
                fig = px.scatter(selected_df, x=numerical[0], y=numerical[1], color=categorical[0], 
                                title=f"Scatter Plot of {numerical[0]} vs {numerical[1]} grouped by {categorical[0]}")
                fig.update_traces(marker=dict(opacity=0.7))
                visualizations.append(dcc.Graph(figure=fig))
        elif plot == "grouped_boxplot":
            create_grouped_boxplot(visualizations, df, numerical, categorical)
        ###several categoric, subgroup
        elif plot == "grouped_barplot":
            if len(selected_vars) == 2:
                fig = create_grouped_bar_plot(selected_df, selected_vars[0], selected_vars[1])
                visualizations.append(dcc.Graph(figure=fig))
        elif plot == "stacked_barplot" and len(selected_vars)==2:
            fig = create_grouped_stacked_bar_plot(selected_df, selected_vars[0], selected_vars[1])
            visualizations.append(dcc.Graph(figure=fig))
        elif plot == "grouped_violin_plot":
            create_grouped_violinplot(visualizations, selected_df, numerical, categorical)
    if not visualizations:
        return html.Div("No valid plots can be generated with the selected variables.", style={"color": "red"})
    return visualizations

if __name__ == "__main__":
    app.run_server(debug=True)
