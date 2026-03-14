import gradio as gr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")


# Weed detection function
def detect_weeds(image):

    results = model(image)

    result_img = results[0].plot()

    boxes = results[0].boxes
    weed_count = len(boxes)

    # spraying recommendation
    spray_level = min(10, weed_count + 1)

    return result_img, weed_count, spray_level


with gr.Blocks(title="Dridha-Agri AI Platform") as demo:

    # --------------------------------------------------
    # PROJECT INTRO
    # --------------------------------------------------

    gr.Markdown(
    """
# 🌱 Dridha-Agri – AI Platform for Smart Agriculture

**Dridha-Agri** is an AI-powered agricultural monitoring platform that uses  
**drone imagery and computer vision** to support **precision agriculture**.

The system integrates:

• 🌿 **Weed detection using YOLO object detection**  
• 🍃 **Plant disease monitoring research**  
• ☁️ **Automatic cloud synchronization of field images**  
• 💧 **Variable rate precision spraying recommendations**

This platform helps farmers **identify crop issues early**,  
**reduce chemical usage**, and **improve crop productivity**.
"""
    )

    # --------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------

    with gr.Tab("📊 Model Overview"):

        gr.Markdown(
        """
## AI Model

**Architecture:** YOLO Object Detection Model

The model analyzes aerial drone images and detects **weed clusters in crop fields**.

### Dataset

Total Images: **3295 annotated agricultural images**

Dataset Split

Train: 85%  
Validation: 10%  
Test: 5%

### Crops Included

• Paddy  
• Cotton  
• Sugarcane  
• Groundnut

Images include **drone aerial imagery and field images collected during crop monitoring**.
"""
        )

    # --------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------

    with gr.Tab("📈 Model Performance"):

        gr.Markdown(
        """
## Evaluation Metrics

| Metric | Score |
|------|------|
| Precision | 90.88% |
| Recall | 85.84% |
| mAP@0.5 | 88.45% |
| mAP@0.5:0.95 | 63.27% |

These results indicate reliable detection of weed clusters in agricultural environments.
"""
        )

    # --------------------------------------------------
    # TRAINING RESULTS CURVE
    # --------------------------------------------------

    with gr.Tab("📉 Training Results"):

        gr.Markdown("## YOLO Training Curve")

        gr.Image(
            value="results.png",
            label="Training Results"
        )

        gr.Markdown(
        """
The training curve shows:

• Precision improvement  
• Recall progression  
• mAP performance  
• Loss convergence

These curves demonstrate stable training and model convergence.
"""
        )

    # --------------------------------------------------
    # LIVE WEED DETECTION
    # --------------------------------------------------

    with gr.Tab("🌿 Live Weed Detection"):

        gr.Markdown(
        """
Upload a **field image** and the AI model will detect weed regions.
"""
        )

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
    # CLOUD SYNCHRONIZATION
    # --------------------------------------------------

    with gr.Tab("☁️ Cloud Synchronization"):

        gr.Markdown(
        """
## Cloud-Based Monitoring System

Images captured by drones or field cameras automatically synchronize to a **cloud platform**.

### Pipeline

Drone Capture → Cloud Upload → AI Analysis → Farmer Dashboard

### Benefits

• Remote farm monitoring  
• Large-scale agricultural data storage  
• Real-time AI analysis  
• Historical crop monitoring
"""
        )

    # --------------------------------------------------
    # VARIABLE RATE SPRAYING
    # --------------------------------------------------

    with gr.Tab("💧 Variable Rate Spraying"):

        gr.Markdown(
        """
## Precision Spraying Strategy

Using weed detection results, the system calculates **spray recommendations**.

| Weed Density | Spray Level |
|--------------|------------|
| Low | 1-3 |
| Medium | 4-7 |
| High | 8-10 |

Instead of spraying chemicals across the entire field,  
**Variable Rate Spraying (VRS)** applies chemicals **only where weeds are detected**.

### Benefits

• Reduce pesticide usage  
• Lower farming costs  
• Protect soil health  
• Improve sustainability
"""
        )

    # --------------------------------------------------
    # PROJECT IMPACT
    # --------------------------------------------------

    with gr.Tab("✅ Project Impact"):

        gr.Markdown(
        """
## Project Achievements

✔ Custom YOLO weed detection model  
✔ Trained on agricultural drone imagery dataset  
✔ Achieved **88% mAP detection accuracy**  
✔ Built an **interactive AI detection interface**  
✔ Designed **cloud-based agricultural monitoring pipeline**  
✔ Implemented **precision spraying recommendations**

## Impact

Dridha-Agri combines:

**Artificial Intelligence + Drone Imagery + Cloud Computing + Precision Agriculture**

to help farmers detect weed infestations early and apply **targeted treatments instead of blanket chemical spraying**.
"""
        )

demo.launch()