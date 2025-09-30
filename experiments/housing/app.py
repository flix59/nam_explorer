import gradio as gr
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nam import NAM, NAM_EXPLANATION, get_shape_function_values, make_nam_architecture_figure
from experiments.housing.train_housing import train_nam
from experiments.housing.dataset import HousingDataset  

# --- Explanation & Architecture figure (add below your two Markdown lines) ---

import matplotlib.pyplot as plt
import numpy as np


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

    def get_shape_function_plot(feature: str):
        x_range, y = values_cell.value[feature]
        plt.figure(figsize=(6, 4))
        plt.plot(x_range, y)
        plt.title(f"Shape Function for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Contribution")
        plt.grid(True)
        return plt.gcf()
    gr.Markdown("### NAM architecture")

    example_inputs = [values_cell.value[feature][0][0] for feature in dataset.features]
    example_outputs = [values_cell.value[feature][0][1] for feature in dataset.features]

    gr.Plot(value=make_nam_architecture_figure(feature_names=dataset.features, example_inputs=example_inputs, example_outputs=example_outputs, task="regression"), label="NAM Architecture")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload your own CSV to train a new NAM model")
            csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
            target_column = gr.Dropdown(choices=dataset.features, value="median_income", label="Target Column Name", interactive=True, max_choices=100)
            train_button = gr.Button("Train NAM Model")
            train_status = gr.Textbox(label="Training Status", interactive=False)

        with gr.Column():
            selected_feature = gr.Dropdown(choices=dataset.features, label="Show Shape Function")
            plot_output = gr.Plot(value=get_shape_function_plot(dataset.features[0]), label="Shape Function Plot")

    data = gr.State(value=None)
    def train_new_model(csv_file, target_column):
        dataset = HousingDataset(csv_file=data.value, target_column=target_column)
        model, model_path = train_nam(Path(csv_file), hidden_dim=128, depth=3)
        train_status.value = f"Model trained and saved to {model_path}"
        values_cell.value = get_shape_function_values(model, dataset.data[dataset.features].values, dataset.features, dataset.scaler)
        plot_output.value = get_shape_function_plot(dataset.features[0])
        return gr.update(choices=dataset.features, value=dataset.features[0]), get_shape_function_plot(dataset.features[0]), train_status
    
    def selected_dataset(csv_file):
        data.value = pd.read_csv(csv_file.name)
        choices = list(data.value.columns)
        return gr.update(choices=choices, value=choices[0] if choices else None)

    csv_file.change(fn=selected_dataset, inputs=[csv_file], outputs=[target_column])
    train_button.click(fn=train_new_model, inputs=[csv_file, target_column], outputs=[selected_feature, plot_output, train_status])
    selected_feature.change(fn=get_shape_function_plot, inputs=[selected_feature], outputs=plot_output)

# Run app
if __name__ == "__main__":
    demo.launch()
