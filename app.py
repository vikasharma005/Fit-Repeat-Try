import os
import cv2
import random
import gradio as gr
from gradio_client import Client

machine_number = 0
model = os.path.join(os.path.dirname(__file__), "models/simon_online/Simon_0.png")
url = os.environ['OA_IP_ADDRESS']
print("API:", url)
client = Client(url)

MODEL_MAP = {
    "AI Model Simon_0": 'models/simon_online/Simon_0.png',
    "AI Model Xuanxuan_0": 'models/xiaoxuan_online/Xuanxuan_0.png',
    "AI Model Yifeng_0": 'models/yifeng_online/Yifeng_0.png'
}


def add_waterprint(img):
    h, w, _ = img.shape
    img = cv2.putText(img, 'AI VTON', (int(0.3 * w), h - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                      (128, 128, 128), 2, cv2.LINE_AA)

    return img


def get_tryon_result(model_name, garment1, garment2, seed=1234):
    # _model = "AI Model " + model_name.split("\\")[-1].split(".")[0]  # windows
    _model = "AI Model " + model_name.split("/")[-1].split(".")[0]  # linux
    print("Use Model:", _model)
    seed = random.randint(0, 1222222222)
    result = client.predict(
        model_name,
        garment1,
        garment2,
        api_name="/get_tryon_result",
        fn_index=seed
    )
    final_img = remove_watermark2(result)
    return final_img


def remove_watermark2(path):
    img = cv2.imread(path)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    y_start = max(h - 45, 0)
    y_end = h
    x_start = max(int(0.3 * w), 0)
    x_end = w

    img_[y_start:y_end, x_start:x_end, :] = [255, 255, 255] 

    return img_


with gr.Blocks(css=".output-image, .input-image, .image-preview {height: 400px !important} ") as demo:
    # gr.Markdown("# Outfit Anyone v0.9")
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div>
            <h1 >Outfit Anyone: Ultra-high quality virtual try-on for Any Clothing and Any Person</h1>
        </div>
        </div>
        """)
    
    with gr.Row():
        with gr.Column():
            init_image = gr.Image(sources='clipboard', type="filepath", label="model", value=model)
            example = gr.Examples(inputs=init_image,
                                  examples_per_page=4,
                                  examples=[
                                            os.path.join(os.path.dirname(__file__), MODEL_MAP.get('AI Model Simon_0')),
                                            os.path.join(os.path.dirname(__file__),
                                                         MODEL_MAP.get('AI Model Xuanxuan_0')),
                                            os.path.join(os.path.dirname(__file__), MODEL_MAP.get('AI Model Yifeng_0')),
                                            ])
        with gr.Column():
            gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                </div>
                </div>
                """)
            with gr.Row():
                garment_top = gr.Image(sources='upload', type="filepath", label="top garment")
                example_top = gr.Examples(inputs=garment_top,
                                          examples_per_page=5,
                                          examples=[os.path.join(os.path.dirname(__file__), "garments/top222.JPG"),
                                                    os.path.join(os.path.dirname(__file__), "garments/top5.png"),
                                                    os.path.join(os.path.dirname(__file__), "garments/top333.png"),
                                                    os.path.join(os.path.dirname(__file__), "garments/dress1.png"),
                                                    os.path.join(os.path.dirname(__file__), "garments/dress2.png"),
                                                    ])
                garment_down = gr.Image(sources='upload', type="filepath", label="lower garment")
                example_down = gr.Examples(inputs=garment_down,
                                           examples_per_page=5,
                                           examples=[os.path.join(os.path.dirname(__file__), "garments/bottom1.png"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom2.PNG"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom3.JPG"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom4.PNG"),
                                                     os.path.join(os.path.dirname(__file__), "garments/bottom5.png"),
                                                     ])

            run_button = gr.Button(value="Run")
        with gr.Column():
            gallery = gr.Image()

            run_button.click(fn=get_tryon_result,
                             inputs=[
                                 init_image,
                                 garment_top,
                                 garment_down,
                             ],
                             outputs=[gallery],
                             concurrency_limit=4)


if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(share=False, server_name='127.0.0.1', server_port=7860)
