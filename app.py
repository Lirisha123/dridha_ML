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


with gr.Blocks(title="Dridha-Agri AI Platform") as demo:

    gr.Markdown("""
# 🌱 Dridha-Agri – AI Weed Detection for Precision Agriculture

From **drone imagery to precision spray prescriptions**,  
Dridha-Agri uses AI to detect weed infestations and enable **targeted pesticide application**.

This reduces chemical usage, farming cost, and environmental impact while improving crop health.
""")

# -----------------------------------------------------
# TAB 1 — OVERVIEW + MODEL PERFORMANCE
# -----------------------------------------------------

    with gr.Tab("📊 Model & Performance"):

        gr.Markdown("""
## 🌾 Project Overview

Dridha-Agri is an **AI-based agricultural monitoring system** that analyzes drone imagery to detect weeds in crop fields.

Instead of spraying pesticides across the entire farm, the system identifies **weed-affected regions** and enables **precision spraying**.

### 🚀 Key Capabilities

• AI-based weed detection using **YOLO object detection**  
• Drone-based crop monitoring  
• Geospatial detection outputs  
• Precision spraying recommendations  
• Reduced chemical usage
""")

        gr.Markdown("---")

        gr.Markdown("""
## 🤖 Model Details

| Feature | Details |
|-------|-------|
| Model | YOLOv11 (Small) |
| Task | Weed Detection from Drone Imagery |
| Dataset | 3295 annotated agricultural images |
| Training | 96 epochs with data augmentation |
| Crops Covered | Paddy 🌾, Cotton 🌿, Sugarcane 🍃, Groundnut 🌱 |
""")

        gr.Markdown("---")

        gr.Markdown("""
## 📈 Model Performance

| Metric | Score | Meaning |
|------|------|------|
| 🎯 Precision | **90.88%** | How many detected weeds are correct |
| 🔍 Recall | **85.84%** | How many real weeds were detected |
| 📊 mAP@0.5 | **88.45%** | Overall detection accuracy |
| 📉 mAP@0.5-0.95 | **63.27%** | Strict accuracy across multiple thresholds |

These metrics show the model can **reliably detect weed clusters in real agricultural environments**.
""")

        gr.Markdown("""
## 🎯 Objective

Accurate weed localization enables **precision spraying**, allowing farmers to treat only affected areas rather than spraying the entire field.
""")

# -----------------------------------------------------
# TAB 2 — RESULT CURVES
# -----------------------------------------------------

    with gr.Tab("📉 Training Results"):

        gr.Markdown("## Training Performance Curves")

        gr.Image(
            value="results.png",
            label="YOLO Training Curves"
        )

        gr.Markdown("""
The training curves illustrate:

• Loss convergence during training  
• Precision and recall improvements  
• mAP performance progression  

These results indicate stable model training and strong detection performance.
""")

# -----------------------------------------------------
# TAB 3 — LIVE DETECTION
# -----------------------------------------------------

    with gr.Tab("🛰 Live Weed Detection"):

        gr.Markdown("## Upload Field Image")

        image_input = gr.Image(type="numpy", label="Upload Drone or Field Image")

        detect_button = gr.Button("Run Weed Detection")

        result_image = gr.Image(label="Detection Result")

        weed_count = gr.Number(label="Detected Weed Count")

        spray_level = gr.Number(label="Recommended Spray Level (1-10)")

        detect_button.click(
            detect_weeds,
            inputs=image_input,
            outputs=[result_image, weed_count, spray_level]
        )

# -----------------------------------------------------
# TAB 4 — WORKFLOW
# -----------------------------------------------------

    with gr.Tab("⚙️ System Workflow"):

        gr.Markdown("""
## End-to-End System Pipeline

Dridha-Agri integrates **drone technology, cloud infrastructure, and AI models** to automate crop monitoring.

### Workflow

1️⃣ Drone missions are planned and executed across agricultural fields.

2️⃣ During flight, high-resolution aerial images are captured.

3️⃣ Images automatically synchronize to the **cloud platform along with geo-referenced data**.

4️⃣ The uploaded imagery is processed by the **AI weed detection model**.

5️⃣ The model identifies weed clusters across crop regions.

6️⃣ Detection results are converted into **geospatial outputs**.

7️⃣ These outputs can guide **precision spraying missions for targeted pesticide application**.

This pipeline enables **large-scale crop monitoring and data-driven farming decisions**.
""")

# -----------------------------------------------------
# TAB 5 — VARIABLE RATE SPRAYING
# -----------------------------------------------------

    with gr.Tab("💧 Variable Rate Spraying"):

        gr.Markdown("""
## Precision Spraying Strategy

Traditional farming sprays pesticides uniformly across entire fields.

Dridha-Agri instead recommends **variable-rate spraying based on weed density**.

### Spray Recommendation Scale

| Weed Density | Spray Level |
|-------------|-------------|
| Low | 1-3 |
| Medium | 4-7 |
| High | 8-10 |

### Benefits

🌿 Reduced chemical usage  
💰 Lower farming costs  
🌎 Reduced environmental impact  
📈 Improved crop productivity
""")

# -----------------------------------------------------
# TAB 6 — CONCLUSION
# -----------------------------------------------------

    with gr.Tab("✅ Conclusion"):

        gr.Markdown("""
## Project Conclusion

Dridha-Agri demonstrates how **AI can directly support precision agriculture**.

In this project we successfully:

• Trained a custom **YOLOv11 model** for weed detection from drone imagery  
• Achieved **~88% mAP@0.5**, ensuring accurate weed localization  
• Enabled **real-time inference suitable for field deployment**  
• Translated detection results into a **1-10 variable rate spraying scale**

This approach reduces unnecessary chemical usage while improving crop health.

### Key Takeaways

✔ AI-driven weed detection is feasible in real agricultural conditions  
✔ Precision spraying can reduce chemical usage by up to **70%**  
✔ Simple, interpretable outputs improve real-world usability  

Dridha-Agri bridges the gap between **computer vision research and practical sustainable farming solutions**.
""")

demo.launch()