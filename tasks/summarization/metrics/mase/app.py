import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mase")
launch_gradio_widget(module)
