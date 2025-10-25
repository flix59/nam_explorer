import gradio as gr
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from nam import NAM, get_shape_function_values
from experiments.housing.components.nam_explanation import NAM_EXPLANATION
from experiments.housing.components.shape_function_plot import make_nam_architecture_figure
from experiments.housing.train_housing import train_nam
from experiments.housing.dataset import HousingDataset


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 NAM Shape Function Explorer example  Housing value prediction.")
    gr.Markdown("Interactively explore Neural Additive Model predictions and shape functions.")

    # Load model and dataset
    pwd = Path(__file__).parent
    dataset = HousingDataset(csv_file=pwd / 'data/housing.csv')
    # --- Explanation & Architecture figure (add below your two Markdown lines) ---

    gr.Markdown(NAM_EXPLANATION)

    model = NAM.load_model(pwd / "models/nam_housing_32_5.pth")
    model.eval()
    values_cell = gr.State(get_shape_function_values(model, dataset.data[dataset.features].values, dataset.features, dataset.scaler))

    def get_all_shape_functions_plot():
        """Combined plot with normalized feature values and contributions [0, 1]"""
        fig = go.Figure()
        # Use keys from values_cell.value to get current features
        for feature in values_cell.value.keys():
            x_range, y = values_cell.value[feature]
            # Normalize x_range to [0, 1]
            x_min, x_max = x_range.min(), x_range.max()
            x_normalized = (x_range - x_min) / (x_max - x_min) if x_max > x_min else x_range
            # Normalize y to [0, 1]
            y_min, y_max = y.min(), y.max()
            y_normalized = (y - y_min) / (y_max - y_min) if y_max > y_min else y
            fig.add_trace(go.Scatter(
                x=x_normalized,
                y=y_normalized,
                mode='lines',
                name=feature,
                hovertemplate=f'{feature}<br>Normalized Value: %{{x:.2f}}<br>Normalized Contribution: %{{y:.2f}}<extra></extra>'
            ))
        fig.update_layout(
            title="Shape Functions for All Features (Normalized)",
            xaxis_title="Normalized Feature Value [0-1]",
            yaxis_title="Normalized Contribution [0-1]",
            hovermode='closest',
            height=600
        )
        return fig

    def get_individual_shape_function_plot(feature):
        """Individual plot for a specific feature with original values"""
        x_range, y = values_cell.value[feature]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y,
            mode='lines',
            name=feature,
            line=dict(width=3),
            hovertemplate=f'Value: %{{x:.2f}}<br>Contribution: %{{y:.2f}}<extra></extra>'
        ))
        fig.update_layout(
            title=f"Shape Function: {feature}",
            xaxis_title=f"{feature} (original scale)",
            yaxis_title="Contribution",
            hovermode='closest',
            height=400,
            showlegend=False
        )
        return fig
    gr.Markdown("### NAM architecture")

    example_inputs = [values_cell.value[feature][0][0] for feature in dataset.features]
    example_outputs = [values_cell.value[feature][0][1] for feature in dataset.features]

    gr.Plot(value=make_nam_architecture_figure(feature_names=dataset.features, example_inputs=example_inputs, example_outputs=example_outputs, task="regression"), label="NAM Architecture")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Shape Functions")
            plot_output = gr.Plot(value=get_all_shape_functions_plot(), label="All Shape Functions (Normalized)")

            gr.Markdown("### Individual Shape Functions (Original Scale)")
            feature_dropdown = gr.Dropdown(choices=dataset.features, value=dataset.features[0], label="Select Feature")
            individual_plot_output = gr.Plot(value=get_individual_shape_function_plot(dataset.features[0]), label="Individual Shape Function")

        with gr.Column(scale=1):
            gr.Markdown("### Upload your own CSV to train a new NAM model")
            csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
            target_column = gr.Dropdown(choices=dataset.features, value="median_income", label="Target Column Name", interactive=True, max_choices=100)
            train_button = gr.Button("Train NAM Model")
            train_status = gr.Textbox(label="Training Status", interactive=False)

    feature_dropdown.change(fn=get_individual_shape_function_plot, inputs=[feature_dropdown], outputs=[individual_plot_output])

    data = gr.State(value=None)
    def train_new_model(csv_file, target_column):
        new_dataset = HousingDataset(csv_file=data.value, target_column=target_column)
        model, model_path = train_nam(Path(csv_file), hidden_dim=32, depth=3, epochs=20)
        train_status_msg = f"Model trained and saved to {model_path}"
        values_cell.value = get_shape_function_values(model, new_dataset.data[new_dataset.features].values, new_dataset.features, new_dataset.scaler)
        # Update dropdown with new features and get first feature's plot
        new_features = list(values_cell.value.keys())
        first_feature = new_features[0] if new_features else None
        return (
            get_all_shape_functions_plot(),
            train_status_msg,
            gr.update(choices=new_features, value=first_feature),
            get_individual_shape_function_plot(first_feature) if first_feature else None
        )

    def selected_dataset(csv_file):
        data.value = pd.read_csv(csv_file.name)
        choices = list(data.value.columns)
        return gr.update(choices=choices, value=choices[0] if choices else None)

    csv_file.change(fn=selected_dataset, inputs=[csv_file], outputs=[target_column])
    train_button.click(fn=train_new_model, inputs=[csv_file, target_column], outputs=[plot_output, train_status, feature_dropdown, individual_plot_output])

# Run app
if __name__ == "__main__":
    demo.launch()
