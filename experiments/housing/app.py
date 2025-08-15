import gradio as gr
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.housing.train_housing import train_nam
from src.nam import NAM
from experiments.housing.dataset import HousingDataset  




# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 NAM Shape Function Explorer example  Housing value prediction.")
    gr.Markdown("Interactively explore Neural Additive Model predictions and shape functions.")
    # Load model and dataset
    pwd = Path(__file__).parent
    dataset = HousingDataset(csv_file=pwd / 'data/housing.csv')
    model = NAM.load_model(pwd / "models/nam_housing_32_5.pth")
    model.eval()
    # Plot shape functions
    def get_shape_function_values(model: NAM, dataset: HousingDataset):
        values = {}
        for index, feature in enumerate(dataset.features):
    
            # Range of values in original space
            x_raw = dataset.data[feature].values
            x_range = np.linspace(x_raw.min(), x_raw.max(), 200)
    
            # Scale to model input space
            x_scaled = dataset.scaler.transform(np.array([x_range if i == index else np.zeros_like(x_range)
                                                  for i in range(len(dataset.features))]).T)

            x_tensor = torch.tensor(x_scaled[:, index].reshape(-1, 1), dtype=torch.float32)
            with torch.no_grad():
                shape_fn = model.shape_functions[index]
                y = shape_fn(x_tensor).view(-1).numpy()
            values[feature] = (x_range, y)
        
        return values
    values_cell = gr.State(get_shape_function_values(model, dataset))

    def get_shape_function_plot(feature: str):
        x_range, y = values_cell.value[feature]
        plt.figure(figsize=(6, 4))
        plt.plot(x_range, y)
        plt.title(f"Shape Function for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Contribution")
        plt.grid(True)
        return plt.gcf()
    test = dataset.features + ["AAAAAAAAAAAAAAAAA"]
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload your own CSV to train a new NAM model")
            csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
            target_column = gr.Dropdown(choices=test, value="median_income", label="Target Column Name", interactive=True, max_choices=100)
            train_button = gr.Button("Train NAM Model")
            train_status = gr.Textbox(label="Training Status", interactive=False)

        with gr.Column():
            selected_feature = gr.Dropdown(choices=test, label="Show Shape Function")
            plot_output = gr.Plot(value=get_shape_function_plot(dataset.features[0]), label="Shape Function Plot")

    data = gr.State(value=None)
    def train_new_model(csv_file, target_column):
        dataset = HousingDataset(csv_file=data.value, target_column=target_column)
        model, model_path = train_nam(Path(csv_file), hidden_dim=5, depth=2)
        train_status.value = f"Model trained and saved to {model_path}"
        values_cell.value = get_shape_function_values(model, dataset)
        plot_output.value = get_shape_function_plot(dataset.features[0])
        return gr.update(choices=dataset.features, value=dataset.features[0]), get_shape_function_plot(dataset.features[0]), train_status
    

    # When a CSV is selected: store its path in state and update the target-column choices
    # def on_csv_selected(file_obj):
    #     if file_obj is None:
    #         return gr.update(),               # target_column (no change)
                
    #     path = file_obj.name
    #     # Read header to populate target choices
    #     cols = list(pd.read_csv(path, nrows=0).columns)
    #     new_value = cols[0] if cols else None
        # return gr.Dropdown.update(choices=cols, value=new_value)
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
