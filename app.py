import gradio as gr
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

def detect_weeds(image):

    results = model(image)
    result_img = results[0].plot()

    boxes = results[0].boxes
    weed_count = len(boxes)

    spray_level = min(10, weed_count + 1)

    return result_img, weed_count, spray_level


with gr.Blocks(title="Dridha-Agri AI System") as demo:

    gr.Markdown(
    """
# 🌱 Dridha-Agri – AI-powered Weed Detection System for Precision Agriculture

From drone imagery to precision spray prescriptions – reducing chemicals, cost, and crop stress using AI.
"""
    )

    # --------------------------------------------------
    # OVERVIEW + MODEL PERFORMANCE
    # --------------------------------------------------

    with gr.Tab("AI Model Performance"):

        gr.Markdown(
        """
## Project Overview

Dridha-Agri is an AI-based agricultural monitoring system designed to support **precision farming using drone imagery and computer vision**.

The system analyzes aerial field images and identifies weed-affected regions so that farmers can apply targeted treatment instead of spraying chemicals across the entire field.

### Key Capabilities

• Automated weed detection using YOLO object detection  
• Drone-based crop monitoring  
• Geospatially aware detection outputs  
• Precision spraying recommendations  
• Reduced chemical usage and improved crop health


---

## YOLOv11 Model – Performance Summary

**Model:** YOLOv11 (Small)  
**Task:** Weed Detection from Drone Imagery  

**Crops Covered**

• Paddy  
• Cotton  
• Sugarcane  
• Groundnut  

**Dataset**

3295 annotated images  
(85% train, 10% validation, 5% test)

**Training**

96 epochs with data augmentation.

**Objective**

Accurate weed localization to enable **precision spraying in agricultural fields**.
"""
        )

    # --------------------------------------------------
    # RESULT CURVES
    # --------------------------------------------------

    with gr.Tab("Result Curves"):

        gr.Markdown("## Training Performance Curves")

        gr.Image(
            value="results.png",
            label="YOLO Training Results"
        )

        gr.Markdown(
        """
These curves show:

• Training and validation loss  
• Precision and recall progression  
• mAP performance improvement during training  

The curves demonstrate stable model convergence.
"""
        )

    # --------------------------------------------------
    # LIVE DETECTION
    # --------------------------------------------------

    with gr.Tab("Live Weed Detection"):

        gr.Markdown("## Upload Image")

        image_input = gr.Image(type="numpy", label="Upload Field Image")

        detect_button = gr.Button("Run Weed Detection")

        result_image = gr.Image(label="Detection Result")

        weed_count = gr.Number(label="Detected Weed Count")

        spray_level = gr.Number(label="Recommended Spray Level (1-10)")

        detect_button.click(
            detect_weeds,
            inputs=image_input,
            outputs=[result_image, weed_count, spray_level]
        )

    # --------------------------------------------------
    # SYSTEM WORKFLOW
    # --------------------------------------------------

    with gr.Tab("System Workflow"):

        gr.Markdown(
        """
## System Workflow

The Dridha-Agri system integrates **drone technology, cloud infrastructure, and AI models** for automated crop monitoring.

1️⃣ Drone missions are planned and executed over agricultural fields.

2️⃣ During flight, high-resolution aerial images of the crop field are captured.

3️⃣ Captured images are automatically synchronized to cloud storage along with geo-referenced metadata.

4️⃣ The uploaded images are processed by the AI weed detection model.

5️⃣ The model analyzes the images and identifies weed clusters within crop regions.

6️⃣ Detected weed locations are translated into geospatial outputs.

7️⃣ These outputs can be used to guide precision spraying operations.

This pipeline enables **large-scale crop monitoring and data-driven agricultural decision making**.
"""
        )

    # --------------------------------------------------
    # VARIABLE RATE SPRAYING
    # --------------------------------------------------

    with gr.Tab("Variable Rate Spraying"):

        gr.Markdown(
        """
## Precision Spraying Strategy

Instead of spraying chemicals across the entire field, Dridha-Agri recommends **variable rate spraying based on weed density**.

### Spray Recommendation Scale

Low Weed Density → Spray Level 1–3  
Medium Weed Density → Spray Level 4–7  
High Weed Density → Spray Level 8–10  

### Benefits

• Reduce pesticide usage  
• Lower operational cost  
• Protect soil and environment  
• Improve crop productivity
"""
        )

    # --------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------

    with gr.Tab("Conclusion"):

        gr.Markdown(
        """
## Conclusion

Dridha-Agri demonstrates how AI can directly support precision agriculture.

In this project, we successfully:

• Trained a custom YOLOv11 model for weed detection from drone imagery  
• Achieved ~88% mAP@0.5 ensuring accurate weed localization  
• Enabled real-time inference suitable for field deployment  
• Translated detection outputs into a **variable rate spraying scale**

This approach reduces unnecessary chemical usage while improving crop health.

### Key Takeaways

✔ AI-driven weed detection is feasible in real agricultural conditions  
✔ Precision spraying can reduce chemical usage significantly  
✔ Simple, interpretable outputs improve real-world usability  

Dridha-Agri bridges the gap between **computer vision research and practical agricultural solutions**.
"""
        )


demo.launch()