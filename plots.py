import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dash import dcc



def create_2d_density_plot(df, x_col, y_col, grid_size=100):
    """
    Generate a 2D density plot (KDE) using a grid-based approach.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        grid_size (int): Resolution of the grid.

    Returns:
        go.Figure: Plotly Figure object for the 2D density plot.
    """
    # Define grid for KDE calculation
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Perform KDE on the grid
    xy_samples = np.vstack([df[x_col], df[y_col]])
    kde = gaussian_kde(xy_samples)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=x_grid,
        y=y_grid,
        colorscale="Blues"
    ))
    fig.update_layout(
        title=f"2D Density Plot of {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    return fig


def create_line_plot_multiple_vars(df, selected_vars):
    """
    Creates a multi-variable line plot with one line highlighted in each subplot.

    Args:
        df (pd.DataFrame): The data frame containing the data.
        selected_vars (list): List of selected variables for the plot (must be > 2).

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    x_var = None
    for var in selected_vars:
        if df[var].is_monotonic_increasing or df[var].is_monotonic_decreasing:
            x_var = var
            break

    if not x_var:
        x_var = df.index
        df = df.reset_index()  # Ensure index is a column if it's used as x_var
        x_var_label = "Index"  # Label for x-axis
    else:
        x_var_label = x_var  # Label for x-axis if an ordered variable is found

    # Remaining variables will be plotted on the y-axis
    y_vars = [var for var in selected_vars if var != x_var]

    # Create subplots with a grid layout
    num_vars = len(y_vars)
    rows = (num_vars - 1) // 3 + 1
    cols = min(num_vars, 3)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=y_vars)
    # Define a color palette
    colors = px.colors.qualitative.Set1  # Use Plotly's Set1 color palette
    num_colors = len(colors)
    # Loop through each y variable
    for i, y_var in enumerate(y_vars):
        row = i // 3 + 1
        col = i % 3 + 1
        line_color = colors[i % num_colors]  # Cycle through the color palette
        # Add grey lines for context
        for other_y_var in y_vars:
            fig.add_trace(
                go.Scatter(
                    x=df[x_var],
                    y=df[other_y_var],
                    mode='lines',
                    line=dict(color='grey', width=1, dash='solid'),
                    opacity=0.3,
                    showlegend=False
                ),
                row=row, col=col
            )
        # Add highlighted line
        fig.add_trace(
            go.Scatter(
                x=df[x_var],
                y=df[y_var],
                mode='lines',
                line=dict(color=line_color, width=3),
                name=y_var,
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text=y_var, row=row, col=col)
    fig.update_layout(
        height=450 * rows, width=1500,
        title_text=f"Comparison of Variables with {x_var} on X-axis",
        title_font_size=16,
        title_x=0.5,
        template="plotly_white"
    )
    fig.update_xaxes(title_text=x_var, row=rows, col=1)
    return fig

def create_line_plot(df, selected_vars):
    """
    Creates a line plot with one ordered variable on the x-axis, if available.

    Args:
        df (pd.DataFrame): The data frame containing the data.
        selected_vars (list): List of selected variables for the plot (must be > 1).

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # Identify an ordered variable
    x_var = None
    for var in selected_vars:
        if df[var].is_monotonic_increasing or df[var].is_monotonic_decreasing:
            x_var = var
            break
    if not x_var:
        raise ValueError("No ordered variable found to use as the x-axis.")
    # Get the first non-x variable as the y-axis variable
    y_var = [var for var in selected_vars if var != x_var][0]
    # Create the line plot
    fig = px.line(
        df, x=x_var, y=y_var,
        title=f"Line Plot of {x_var} vs {y_var}",
        template="plotly_white"  # Ensures white background and clean gridlines
    )
    return fig

def create_pairplot(visualizations, df, numerical, categorical):
    """
    Create a pair plot for the selected variables, optionally grouping by a categorical variable.

    Args:
        visualizations (list): List to append the visualization graph.
        df (pd.DataFrame): The data frame containing the data.
        numerical (list): List of numerical variables for the pair plot.
        categorical (list): List of categorical variables for coloring (should be empty or contain one item).
    """
    selected_vars = numerical + (categorical if categorical else [])
    # Create a pair plot using Plotly Express
    dimensions = [{"label": col, "values": df[col]} for col in numerical]
    
    if categorical:
        cat_var = categorical[0]
        # Encode categorical variable as integers for coloring
        df_encoded = df.copy()
        df_encoded[cat_var] = df_encoded[cat_var].astype('category').cat.codes
        colors = df_encoded[cat_var]
        category_mapping = dict(enumerate(df[cat_var].astype('category').cat.categories))

        fig = go.Figure(data=go.Splom(
            dimensions=dimensions,
            showupperhalf=False,  # Only show lower triangle of the matrix
            diagonal_visible=True,  # Include the diagonal
            marker=dict(
                size=5,  # Marker size
                line=dict(width=0.5, color='white'),  # Marker border
                color=colors,  # Numerical encoding for coloring
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=cat_var,
                    tickvals=list(category_mapping.keys()),
                    ticktext=list(category_mapping.values())
                )
            )
        ))
    
    else:
        fig = go.Figure(data=go.Splom(
            dimensions=dimensions,
            showupperhalf=False,  # Only show lower triangle of the matrix
            diagonal_visible=True,  # Include the diagonal
            marker=dict(
                size=5,  # Marker size
                line=dict(width=0.5, color='white')  # Marker border
            )
        ))

    # Customize layout
    fig.update_layout(
        title="Scatterplot Matrix (Correlogram)",
        dragmode="select",  # Enable selection
        width=900,
        height=900,
        template="plotly_white"
    )
    visualizations.append(dcc.Graph(figure=fig))

def create_dendrogram_with_heatmap(dataframe, colorscale="Blues"):
    """
    Create a dendrogram with a heatmap for a given DataFrame.
    
    Parameters:
        dataframe (pd.DataFrame): DataFrame with numeric values to cluster.
        colorscale (str): Color scale for the heatmap. Default is 'Blues'.
    
    Returns:
        go.Figure: Plotly figure object containing the dendrogram and heatmap.
    """
    # Convert DataFrame to a numpy array
    data_array = dataframe.to_numpy().T
    labels = dataframe.columns.tolist()

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(data_array, orientation='bottom', labels=labels)
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(data_array, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = list(map(int, dendro_side['layout']['yaxis']['ticktext']))
    data_dist = pdist(data_array)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]

    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale=colorscale,
        )
    ]

    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout({
        'width': 1000,
        'height': 1000,
        'showlegend': False,
        'hovermode': 'closest',
    })
    
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks': ""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks': ""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""
                            })
    # Edit yaxis2
    fig.update_layout(yaxis2={'domain': [.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks': ""})

    return fig

def create_PCA_scatter_plot(visualizations, df, num_vars, cat_vars):
    # Case: More than two numerical variables and one categorical variable
    if len(num_vars) > 2 and len(cat_vars) == 1:
        cat_var = cat_vars[0]
        df_scaled = StandardScaler().fit_transform(df[num_vars])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_scaled)
        
        # Explained Variance
        explained_variance = pca.explained_variance_ratio_
        
        # Create a DataFrame for the components
        pca_df = pd.DataFrame(
            data=principal_components, columns=['PC1', 'PC2']
        )
        pca_df['Sample'] = df.index
        pca_df[cat_var] = df[cat_var]
        
        # Plot the PCA results using Plotly
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color=cat_var,
            title=f"PCA Scatter Plot ({explained_variance[0] * 100:.1f}% + {explained_variance[1] * 100:.1f}% Variance)",
            labels={
                'PC1': f"PC1 ({explained_variance[0] * 100:.1f}% Variance)",
                'PC2': f"PC2 ({explained_variance[1] * 100:.1f}% Variance)"
            },
            hover_data={'Sample': True, 'PC1': False, 'PC2': False}  # Display sample on hover only
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.8), opacity=0.6))
        fig.update_layout(
            xaxis=dict(title=f"PC1 ({explained_variance[0] * 100:.1f}% Variance)"),
            yaxis=dict(title=f"PC2 ({explained_variance[1] * 100:.1f}% Variance)"),
            template="plotly_white", width=800, height=600
        )
        visualizations.append(dcc.Graph(figure=fig))

    # Case: More than two numerical variables and no categorical variable
    elif len(num_vars) > 2 and len(cat_vars) == 0:
        df_scaled = StandardScaler().fit_transform(df[num_vars])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_scaled)
        
        # Explained Variance
        explained_variance = pca.explained_variance_ratio_
        
        # Create a DataFrame for the components
        pca_df = pd.DataFrame(
            data=principal_components, columns=['PC1', 'PC2']
        )
        pca_df['Sample'] = df.index
        
        # Plot the PCA results using Plotly
        fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            title=f"PCA Scatter Plot ({explained_variance[0] * 100:.1f}% + {explained_variance[1] * 100:.1f}% Variance)",
            labels={
                'PC1': f"PC1 ({explained_variance[0] * 100:.1f}% Variance)",
                'PC2': f"PC2 ({explained_variance[1] * 100:.1f}% Variance)"
            },
            hover_data={'Sample': True, 'PC1': False, 'PC2': False}  # Display sample on hover only
        )
        fig.update_traces(marker=dict(size=8, color='#2fcee1', line=dict(width=0.8, color='#246d75'), opacity=0.6))
        fig.update_layout(
            xaxis=dict(title=f"PC1 ({explained_variance[0] * 100:.1f}% Variance)"),
            yaxis=dict(title=f"PC2 ({explained_variance[1] * 100:.1f}% Variance)"),
            showlegend=False, 
            template="plotly_white", width=800, height=600
        )
        visualizations.append(dcc.Graph(figure=fig))

    return visualizations

def create_lollipop_plot(visualizations, df, num_vars, cat_vars):
    # Case: One categorical variable
    if len(cat_vars) == 1 and len(num_vars) == 0:
        cat_col = cat_vars[0]
        fig = create_lollipop_chart(df, cat_col)
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One numerical variable and one categorical variable
    elif len(cat_vars) == 1 and len(num_vars) == 1:
        cat_col = cat_vars[0]
        num_col = num_vars[0]
        fig = create_lollipop_chart(df, cat_col, num_col)
        visualizations.append(dcc.Graph(figure=fig))

    return visualizations

def create_lollipop_chart(df, cat_col, num_col=None):
    if num_col is None:
        # Case: One categorical variable
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, 'Count']

        data = [
            go.Scatter(
                y=counts['Count'],
                x=counts[cat_col],
                mode='markers',
                marker=dict(color='red')
            )
        ]
        # Use the 'shapes' attribute from the layout to draw the vertical lines
        layout = go.Layout(
            shapes=[dict(
                type='line',
                xref='x',
                yref='y',
                y0=counts['Count'][i],
                x0=counts[cat_col][i],
                y1=0,
                x1=counts[cat_col][i],
                line=dict(
                    color='grey',
                    width=1
                )
            ) for i in range(len(counts))],
            title=f'Lollipop Chart of {cat_col}',
            xaxis_title=cat_col,
            yaxis_title='Count'
        )

    else:
        # Case: One numerical variable and one categorical variable
        agg_data = df.groupby(cat_col)[num_col].median().reset_index()
        agg_data.columns = [cat_col, num_col]

        data = [
            go.Scatter(
                y=agg_data[num_col],
                x=agg_data[cat_col],
                mode='markers',
                marker=dict(color='blue')
            )
        ]
        # Use the 'shapes' attribute from the layout to draw the vertical lines
        layout = go.Layout(
            shapes=[dict(
                type='line',
                xref='x',
                yref='y',
                y0=agg_data[num_col][i],
                x0=agg_data[cat_col][i],
                y1=0,
                x1=agg_data[cat_col][i],
                line=dict(
                    color='grey',
                    width=1
                )
            ) for i in range(len(agg_data))],
            title=f'Lollipop Chart of {num_col} by {cat_col}',
            xaxis_title=cat_col,
            yaxis_title=f'Median {num_col}'
        )
    fig = go.Figure(data, layout)
    return fig

def create_sorted_bar_plot(df, categorical_col): # sorted from highest count to lowest
    # Count the occurrences of each category and sort in descending order
    sorted_df = df[categorical_col].value_counts().reset_index()
    sorted_df.columns = [categorical_col, 'count']  # Rename columns for clarity
    sorted_df = sorted_df.sort_values(by='count', ascending=False)
    # Create the bar plot with sorted values
    fig = px.bar(sorted_df, 
                 x=categorical_col, 
                 y='count', 
                 title=f"Bar Plot of {categorical_col} (Ordered by Frequency)",
                 text='count')  # Add text labels for counts
    # Update layout for cleaner visualization
    fig.update_traces(textposition='outside')  # Place text labels outside the bars
    fig.update_layout(xaxis_title=categorical_col, 
                      yaxis_title="Count", 
                      uniformtext_minsize=8, 
                      uniformtext_mode='hide')
    return fig

def create_grouped_bar_plot(df, col1, col2):
    # Determine which column has fewer unique values
    unique_vals_col1 = df[col1].nunique()
    unique_vals_col2 = df[col2].nunique()

    if unique_vals_col1 <= unique_vals_col2:
        grouping_col = col2
        value_col = col1
    else:
        grouping_col = col1
        value_col = col2

    # Count occurrences and create a grouped bar plot
    grouped_df = df.groupby([grouping_col, value_col]).size().reset_index(name='count')

    fig = px.bar(grouped_df, 
                 x=grouping_col, 
                 y='count', 
                 color=value_col, 
                 barmode='group',
                 title=f"Grouped Bar Plot of {grouping_col} and {value_col}")

    fig.update_layout(xaxis_title=grouping_col, 
                      yaxis_title="Count", 
                      uniformtext_minsize=8, 
                      uniformtext_mode='hide')

    return fig

def create_grouped_stacked_bar_plot(df, col1, col2):
    # Determine which column has fewer unique values
    unique_vals_col1 = df[col1].nunique()
    unique_vals_col2 = df[col2].nunique()

    if unique_vals_col1 <= unique_vals_col2:
        grouping_col = col2
        value_col = col1
    else:
        grouping_col = col1
        value_col = col2

    # Count occurrences and pivot the DataFrame for stacking
    grouped_df = df.groupby([grouping_col, value_col]).size().reset_index(name='count')
    pivot_df = grouped_df.pivot(index=grouping_col, columns=value_col, values='count').fillna(0)


    # Create the stacked bar plot
    fig = px.bar(pivot_df, 
                 x=pivot_df.index, 
                 y=pivot_df.columns,
                 title=f"Grouped Stacked Bar Plot of {grouping_col} and {value_col}",
                 labels={'value': 'Count', 'variable': value_col})

    fig.update_layout(xaxis_title=grouping_col, 
                      yaxis_title="Count", 
                      barmode='stack',
                      uniformtext_minsize=8, 
                      uniformtext_mode='hide')

    return fig


def create_boxplot(visualizations, selected_df, num_vars, cat_vars):
    # Case: One numerical variable
    if len(num_vars) == 1 and len(cat_vars) == 0:
        fig = px.box(selected_df, y=num_vars[0], title=f"Boxplot of {num_vars[0]}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: Several numerical variables
    elif len(num_vars) > 1 and len(cat_vars) == 0:
        fig = px.box(selected_df, y=num_vars, title=f"Boxplot of {', '.join(num_vars)}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One categorical and one numerical variable
    elif len(cat_vars) == 1 and len(num_vars) == 1:
        fig = px.box(selected_df, x=cat_vars[0], y=num_vars[0], title=f"Boxplot of {num_vars[0]} by {cat_vars[0]}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One categorical variable and several numerical variables
    if len(cat_vars) == 1 and len(num_vars) > 1:
        # Reshape the data so that all numerical variables are stacked under one column
        melted_df = selected_df.melt(id_vars=cat_vars, value_vars=num_vars, 
                                     var_name='Variable', value_name='Value')

        # Create the boxplot with color by the 'Variable' column
        fig = px.box(melted_df, x="Variable", y="Value", color=cat_vars[0],
                     title="Boxplot of Numerical Variables by " + cat_vars[0])
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    return visualizations


def create_violin_plot(visualizations, selected_df, num_vars, cat_vars):

    # Case: One categorical variable and one numerical variable
    if len(cat_vars) == 1 and len(num_vars) == 1:
        fig = px.violin(selected_df, x=cat_vars[0], y=num_vars[0], box=True, points="all", 
                        title=f"Violin Plot of {num_vars[0]} by {cat_vars[0]}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One categorical variable and several numerical variables
    elif len(cat_vars) == 1 and len(num_vars) > 1:
        # Reshape the data so that all numerical variables are stacked under one column
        melted_df = selected_df.melt(id_vars=cat_vars, value_vars=num_vars, 
                                     var_name='Variable', value_name='Value')
        show_points = "all" if len(num_vars) <= 2 else False
        # Create the violin plot with color by the 'Variable' column
        fig = px.violin(melted_df, x="Variable", y="Value", color=cat_vars[0], 
                        box=True, points=show_points, 
                        title=f"Violin Plot of Numerical Variables by {cat_vars[0]}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One numerical variable and several categorical variables
    elif len(cat_vars) > 1 and len(num_vars) == 1:
        # Reshape the data for multiple categories
        melted_df = selected_df.melt(id_vars=cat_vars, value_vars=num_vars, 
                                     var_name='Variable', value_name='Value')
        show_points = "all" if len(cat_vars) <= 2 else False
        # Create a violin plot where categorical variables are on the x-axis
        fig = px.violin(melted_df, x="Variable", y="Value", color="Variable", 
                        box=True, points=show_points, 
                        title=f"Violin Plot of {num_vars[0]} by Categorical Variables")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: Multiple categorical and multiple numerical variables
    elif len(cat_vars) > 1 and len(num_vars) > 1:
        # Reshape the data so that each numerical variable is represented with 'Variable' column
        melted_df = selected_df.melt(id_vars=cat_vars, value_vars=num_vars, 
                                     var_name='Variable', value_name='Value')
        
        show_points = "all" if (len(num_vars)+ len(cat_vars)) <= 2 else False
        # Create the violin plot with color by the 'Variable' column
        fig = px.violin(melted_df, x="Variable", y="Value", color="Variable", 
                        box=True, points=show_points, 
                        title="Violin Plot of Numerical Variables by Categorical Variables")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    
    # Case: Multiple numerical variables
    elif len(num_vars) > 1:
        fig = go.Figure()

        for col in num_vars:
            fig.add_trace(go.Violin(
                y=selected_df[col],
                name=col,
                box_visible=True,  # Show box plot inside the violin
                meanline_visible=True,  # Show mean line
                line_color='black',
                fillcolor='lightseagreen',  #636EFA
                opacity=0.6
            ))

        fig.update_layout(
            title=f"Violin Plot of {', '.join(num_vars)}",
            yaxis_zeroline=False,
            template="plotly_white",
            height=650,
            showlegend=False 
        )

        visualizations.append(dcc.Graph(figure=fig))


    """# Case: Multiple numerical variables
    elif len(num_vars) > 1:
        # Show points only if there are 4 or fewer numerical variables
        show_points = "all" if len(num_vars) <= 2 else False
        fig = px.violin(selected_df, y=num_vars, points=show_points, box=True, 
                        title=f"Violin Plot of {', '.join(num_vars)}")
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))"""

    return visualizations

def create_ridgeline_plot(visualizations, df, num_vars, cat_vars):

    # Case: One categorical and one numerical variable
    if len(cat_vars) == 1 and len(num_vars) == 1:
        fig = px.violin(df, x=num_vars[0], y=cat_vars[0], color=cat_vars[0], 
                        orientation='h', points=False,
                        title=f"Ridgeline Plot of {num_vars[0]} by {cat_vars[0]}")
        fig.update_traces(side='positive', width=3) # makes it a ridgeline plot
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: Multiple numerical variables
    elif len(cat_vars) == 0 and len(num_vars) > 3:
        melted_df = df.melt(value_vars=num_vars, var_name='Variable', value_name='Value')
        
        # Create a ridgeline plot (density plot stacked vertically)
        fig = px.violin(
            melted_df,
            x='Value',
            y='Variable',
            color='Variable',
            orientation='h',
            points=False,  # No points
            #box=True,  # Show box plot inside the violins
            hover_data=melted_df.columns,
            title="Ridgeline Plot of Numerical Variables"
        )
        
        fig.update_traces(side='positive', width=3)  # Customize appearance
        fig.update_layout(height=650, template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))
    return visualizations


def create_kde_plot(visualizations, selected_df, num_vars, cat_vars):
    # Case: Single numerical variable
    if len(num_vars) == 1 and (cat_vars is None or len(cat_vars) == 0):
        col = num_vars[0]
        fig = ff.create_distplot(
            [selected_df[col]], 
            [col], 
            show_hist=False,  # No histogram, only the density curve
            show_rug=False  # No rug plot
        )
        fig.update_layout(
            title=f"Density Plot of {col}",
            template="plotly_white"
        )
        visualizations.append(dcc.Graph(figure=fig))

    # Case: Two or more numerical variables
    elif len(num_vars) > 1 and (len(cat_vars) == 0): 
        data = [selected_df[col] for col in num_vars]  # List of data for each variable
        labels = num_vars  # Labels for legend
        
        fig = ff.create_distplot(
            data, 
            labels, 
            show_hist=False,  # No histogram
            show_rug=False  # No rug plot
        )
        fig.update_layout(
            title=f"Density Plot of {', '.join(num_vars)}",
            template="plotly_white"
        )
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One numerical variable and one categorical variable
    elif len(num_vars) == 1 and len(cat_vars) == 1:
        num_col = num_vars[0]
        cat_col = cat_vars[0]

        # Check the number of unique values in the categorical variable
        unique_values = selected_df[cat_col].nunique()

        if unique_values > 4:
            # Create a subplot figure with density plots for each category
            unique_cats = selected_df[cat_col].unique()
            num_plots = len(unique_cats)
            rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate the number of rows needed
            fig = make_subplots(rows=rows, cols=3, shared_xaxes=True, subplot_titles=[f"{cat_col}: {cat}" for cat in unique_cats])

            for i, cat_value in enumerate(unique_cats):
                filtered_df = selected_df[selected_df[cat_col] == cat_value]
                 # Check if filtered_df has more than one element
                if len(filtered_df[num_col]) > 1:
                    density_fig = ff.create_distplot(
                        [filtered_df[num_col]], 
                        [f"{cat_value}"], 
                        show_hist=False,  # No histogram
                        show_rug=False  # No rug plot
                    )
                    
                    row = (i // 3) + 1
                    col = (i % 3) + 1
                    for trace in density_fig['data']:
                        fig.add_trace(trace, row=row, col=col)
                else:
                    print(f"Skipping category '{cat_value}' as it does not have enough data points.")
            
            fig.update_layout(
                title=f"Density Plot of {num_col} Grouped by {cat_col} (Facet Grid)",
                template="plotly_white"
            )
            visualizations.append(dcc.Graph(figure=fig))
        else:
            groups = selected_df[cat_col].unique()
            data = [selected_df[selected_df[cat_col] == group][num_col] for group in groups]
            labels = [str(group) for group in groups]  # Convert labels to strings

            fig = ff.create_distplot(
                data, 
                labels, 
                show_hist=False,  # No histogram
                show_rug=False,  # No rug plot
            )
            fig.update_layout(
                title=f"Density Plot of {num_col} Grouped by {cat_col}",
                template="plotly_white"
            )
            visualizations.append(dcc.Graph(figure=fig))

    return visualizations


def create_histogram_plot(visualizations, selected_df, num_vars, cat_vars):
    # Case: Single numerical variable
    if len(num_vars) == 1 and (cat_vars is None or len(cat_vars) == 0):
        col = num_vars[0]
        fig = px.histogram(selected_df, x=col, title=f"Histogram of {col}")
        fig.update_layout(template="plotly_white")
        visualizations.append(dcc.Graph(figure=fig))

    # Case: Two numerical variables
    elif len(num_vars) == 2 and (cat_vars is None or len(cat_vars) == 0): 
        melted_df = selected_df[num_vars].melt(var_name="Variable", value_name="Value")
        fig = px.histogram(
            melted_df,
            x="Value",
            color="Variable",  # Differentiates the two variables by color
            barmode="overlay",  # Overlays the histograms
            title=f"Histogram of {num_vars[0]} and {num_vars[1]}"
        )
        fig.update_layout(
            bargap=0.1,  # Adjust gap between bars
            template="plotly_white"
        )
        visualizations.append(dcc.Graph(figure=fig))

    # Case: One numerical variable and one categorical variable
    elif len(num_vars) == 1 and len(cat_vars) == 1:
        num_col = num_vars[0]
        cat_col = cat_vars[0]

        # Check the number of unique values in the categorical variable
        unique_values = selected_df[cat_col].nunique()

        if unique_values > 2:
            # Create a subplot figure with histograms for each category
            unique_cats = selected_df[cat_col].unique()
            num_plots = len(unique_cats)
            rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate the number of rows needed
            fig = make_subplots(rows=rows, cols=3, shared_yaxes=True, subplot_titles=[f"{cat_col}: {cat}" for cat in unique_cats])

            for i, cat_value in enumerate(unique_cats):
                filtered_df = selected_df[selected_df[cat_col] == cat_value]
                hist_fig = px.histogram(filtered_df, x=num_col, title=f"Histogram of {num_col} for {cat_value}")
                hist_fig.update_traces(name=f"{cat_value}")
                row = (i // 3) + 1
                col = (i % 3) + 1
                for trace in hist_fig['data']:
                    fig.add_trace(trace, row=row, col=col)

            fig.update_layout(
                title=f"Histogram of {num_col} Grouped by {cat_col} (Facet Grid)",
                template="plotly_white",
                height = 250 * rows
            )
            visualizations.append(dcc.Graph(figure=fig))
        else:
            fig = px.histogram(
                selected_df, 
                x=num_col, 
                color=cat_col, 
                barmode="overlay", 
                title=f"Histogram of {num_col} Grouped by {cat_col}"
            )
            fig.update_layout(
                bargap=0.1,  # Adjust gap between bars
                template="plotly_white"
            )
            visualizations.append(dcc.Graph(figure=fig))

    return visualizations


def create_grouped_boxplot(visualizations, df, numerical, categorical):
    """
    Create a grouped boxplot for the selected variables, optionally grouping by one or two categorical variables.

    Args:
        visualizations (list): List to append the visualization graph.
        df (pd.DataFrame): The data frame containing the data.
        numerical (list): List of numerical variables for the boxplot.
        categorical (list): List of categorical variables for grouping (should contain one or two items).
    """
    if len(numerical) == 1 and len(categorical) == 2:
        num_var = numerical[0]
        cat_vars = sorted(categorical, key=lambda col: df[col].nunique())
        x_var = cat_vars[1]  # Categorical variable with more unique values
        color_var = cat_vars[0]  # Categorical variable with fewer unique values

        fig = px.box(df, x=x_var, y=num_var, color=color_var, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(
            title=f"Box Plot of {num_var} by {x_var} and {color_var}",
            xaxis_title=x_var,
            yaxis_title=num_var,
            legend_title=color_var,
            template="plotly_white"
        )
        visualizations.append(dcc.Graph(figure=fig))

    elif len(numerical) == 1 and len(categorical) == 3:
        num_var = numerical[0]
        cat0 = df[categorical[0]].unique()
        cat1 = df[categorical[1]].unique()
        third_cat_var = categorical[2]

        fig = make_subplots(
            rows=1, 
            cols=len(cat0), 
            subplot_titles=[f"{categorical[0]}: {b}" for b in cat0],
            shared_yaxes=True
        )

        color_map = px.colors.qualitative.Set1  # Choose a color palette
        color_dict = {treatment: color_map[i % len(color_map)] for i, treatment in enumerate(cat1)}

        for i, firstcat in enumerate(cat0):
            filtered_data = df[df[categorical[0]] == firstcat]
            for cat in cat1:
                cat1_data = filtered_data[filtered_data[categorical[1]] == cat]
                fig.add_trace(
                    go.Box(
                        x=cat1_data[third_cat_var] + " - " + cat1_data[categorical[1]],  # Combine third_cat_var and second categorical for grouping
                        y=cat1_data[num_var],
                        name=f"{cat}",  # Label for the legend
                        marker_color=color_dict[cat],  # Use category-specific color
                        showlegend=True  # Show legend only for the first subplot
                    ),
                    row=1,
                    col=i + 1
                )

        fig.update_layout(
            title=f"Grouped Box Plot of {num_var} by {categorical[0]}, {categorical[1]}, and {third_cat_var}",
            xaxis_title=f"{third_cat_var} and {categorical[1]}",
            yaxis_title=num_var,
            legend_title=categorical[1],
            height=500,
            width=2200,
            template="plotly_white"
        )
        fig.update_xaxes(tickangle=45)
        visualizations.append(dcc.Graph(figure=fig))


def create_grouped_violinplot(visualizations, df, numerical, categorical):
    """
    Create a grouped violin plot for the selected variables, optionally grouping by one or two categorical variables.

    Args:
        visualizations (list): List to append the visualization graph.
        df (pd.DataFrame): The data frame containing the data.
        numerical (list): List of numerical variables for the violin plot.
        categorical (list): List of categorical variables for grouping (should contain two or three items).
    """
    # Ensure categorical variables are treated as strings
    for cat_var in categorical:
        if not df[cat_var].dtype == 'O':  # Check if it's not already a string (dtype 'O' represents 'object' which is string in pandas)
            df[cat_var] = df[cat_var].astype(str)

    if len(numerical) == 1 and len(categorical) == 2:
        num_var = numerical[0]
        cat0, cat1 = categorical

        fig = go.Figure()
        unique_cat0 = df[cat0].unique()
        unique_cat1 = df[cat1].unique()

        color_map = px.colors.qualitative.Set1
        color_dict = {cat: color_map[i % len(color_map)] for i, cat in enumerate(unique_cat0)}

        for val1 in unique_cat1:
            for val0 in unique_cat0:
                filtered_data = df[(df[cat1] == val1) & (df[cat0] == val0)]
                side = "negative" if val0 == unique_cat0[0] else "positive"
                color = color_dict[val0]
                
                fig.add_trace(go.Violin(
                    x=filtered_data[cat1],
                    y=filtered_data[num_var],
                    name=f"{val1} - {val0}",
                    legendgroup=val0,
                    scalegroup=val1,
                    side=side,
                    line_color=color,
                    width=1,
                ))

        fig.update_traces(meanline_visible=True)
        fig.update_layout(
            violingap=0.01,
            violinmode="group",
            title=f"Split Violin Plot of {num_var} by {cat0} and {cat1}",
            xaxis_title=cat1,
            yaxis_title=num_var,
            legend_title=cat0,
            height=600,
            width=800,
            template="plotly_white",
        )
        visualizations.append(dcc.Graph(figure=fig))

    elif len(numerical) == 1 and len(categorical) == 3:
        num_var = numerical[0]
        cat0, cat1, cat2 = categorical

        unique_cat0 = df[cat0].unique()
        unique_cat1 = df[cat1].unique()
        unique_cat2 = df[cat2].unique()

        fig = make_subplots(
            rows=1, 
            cols=len(unique_cat0), 
            subplot_titles=[f"{cat0}: {val}" for val in unique_cat0],
            shared_yaxes=True
        )

        color_map = px.colors.qualitative.Set1
        color_dict = {cat: color_map[i % len(color_map)] for i, cat in enumerate(unique_cat1)}
        
        seen_categories = set()

        for i, val0 in enumerate(unique_cat0):
            filtered_data = df[df[cat0] == val0]
            for val1 in unique_cat1:
                for val2 in unique_cat2:
                    cat_data = filtered_data[(filtered_data[cat1] == val1) & (filtered_data[cat2] == val2)]
                    legend_show = True if val1 not in seen_categories else False
                    seen_categories.add(val1)
                    
                    fig.add_trace(
                        go.Violin(
                            x=cat_data[cat2] + " - " + cat_data[cat1],
                            y=cat_data[num_var],
                            name=f"{val1}",
                            legendgroup=val1,
                            scalegroup=val0,
                            line_color=color_dict[val1],
                            width=1,
                            side="positive",
                            showlegend=legend_show
                        ),
                        row=1,
                        col=i + 1
                    )
        fig.update_traces(meanline_visible=True)
        fig.update_layout(
            title=f"Grouped Violin Plot of {num_var} by {cat0}, {cat1}, and {cat2}",
            xaxis_title=f"{cat2} and {cat1}",
            yaxis_title=num_var,
            legend_title=cat1,
            height=600,
            width=300 * len(unique_cat0),
            template="plotly_white",
        )
        fig.update_xaxes(tickangle=45)
        visualizations.append(dcc.Graph(figure=fig))


