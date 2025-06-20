import gradio as gr

class MainWidget:
    def __init__(self):
        with gr.Blocks(title="深度学习图像处理") as self.demo:
            with gr.Tabs():
                pass


    def launch(self):
        self.demo.launch(server_port=7961)

# Launch the app
if __name__ == "__main__":
    MainWidget().launch()