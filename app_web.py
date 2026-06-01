import gradio as gr
import cv2
import numpy as np
import sys
from gtts import gTTS
import os

sys.path.append('backend')
import inference

def process_image(image):
    if image is None:
        return None, "Please upload an image.", None
        
    # Convert Gradio image (RGB numpy array) to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run our hybrid inference pipeline
    result = inference.run_inference(img_bgr)
    text_output = result.get('text', '')
    
    # Draw boxes
    annotated_bgr = inference.draw_results(img_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate TTS audio
    audio_path = None
    if text_output.strip():
        try:
            tts = gTTS(text=text_output, lang='en')
            audio_path = "temp_audio.mp3"
            tts.save(audio_path)
        except Exception as e:
            print("TTS Error:", e)
            
    return annotated_rgb, text_output, audio_path

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ BrailleVision v2.0 - Live Web App")
    gr.Markdown("Real-time Optical Braille Recognition with Hybrid NLP. Point your phone camera at Braille or upload an image!")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Braille Image or Take Photo")
            submit_btn = gr.Button("Translate Braille", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detected Cells")
            output_text = gr.Textbox(label="Translated Text (with NLP autocorrect)")
            output_audio = gr.Audio(label="Spoken Translation", type="filepath")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_text, output_audio]
    )

if __name__ == "__main__":
    demo.launch(share=True)
