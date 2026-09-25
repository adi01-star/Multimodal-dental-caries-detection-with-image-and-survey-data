"""
gradio-app.py
-------------
Web UI with three tabs, matching the project's three evaluation levels:

  1. Image Only      - upload a dental photo, run the YOLO pipeline.
  2. Survey Only      - answer lifestyle/behavioral questions, run the
                        XGBoost/Random Forest/Logistic Regression pipeline.
  3. Combined         - do both, and see the late-fusion result.

This is a screening / risk-assessment demo, not a diagnostic tool - the UI
says so explicitly.
"""

import gradio as gr

from predict import predict
from fusion import combine
from lifestyle_model import load_schema

try:
    from lifestyle_predict import predict_lifestyle_risk

    LIFESTYLE_AVAILABLE = True
    LIFESTYLE_ERROR = None
except FileNotFoundError as e:
    # Lifestyle model hasn't been trained yet - the Image Only tab should
    # still work, so we degrade gracefully instead of crashing the app.
    LIFESTYLE_AVAILABLE = False
    LIFESTYLE_ERROR = str(e)

DISCLAIMER = (
    "⚠️ This tool is a **screening / risk-assessment aid**, not a clinical "
    "diagnosis. Always confirm with a licensed dentist."
)


# ---------------------------------------------------------------------------
# Tab 1: Image only
# ---------------------------------------------------------------------------
def gradio_predict_image(image: gr.Image):
    img_pil = image.convert("RGB")
    img_with_boxes, status, confidence = predict(img_pil)
    return img_with_boxes, f"{status} (confidence: {confidence:.2f})"


# ---------------------------------------------------------------------------
# Tab 2: Survey / lifestyle only
# ---------------------------------------------------------------------------
def _build_survey_inputs():
    """Builds Gradio inputs dynamically from data/feature_schema.json so the
    form always matches whatever the model was trained on."""
    schema = load_schema()
    components = []
    component_names = []

    for spec in schema["numeric_features"]:
        components.append(
            gr.Slider(
                minimum=spec["min"],
                maximum=spec["max"],
                value=spec["default"],
                step=1 if spec["max"] - spec["min"] >= 5 else 0.5,
                label=spec["label"],
            )
        )
        component_names.append(spec["name"])

    for spec in schema["categorical_features"]:
        components.append(
            gr.Dropdown(choices=spec["choices"], value=spec["default"], label=spec["label"])
        )
        component_names.append(spec["name"])

    return components, component_names


def gradio_predict_survey(*values):
    if not LIFESTYLE_AVAILABLE:
        return f"Lifestyle model not available: {LIFESTYLE_ERROR}"

    answers = dict(zip(_SURVEY_FIELD_NAMES, values))
    probability, status, _details = predict_lifestyle_risk(answers)
    return f"{status} (probability: {probability:.2f})"


# ---------------------------------------------------------------------------
# Tab 3: Combined (late fusion)
# ---------------------------------------------------------------------------
def gradio_predict_combined(image: gr.Image, image_weight: float, *values):
    if not LIFESTYLE_AVAILABLE:
        return None, f"Lifestyle model not available: {LIFESTYLE_ERROR}"

    img_pil = image.convert("RGB")
    img_with_boxes, image_status, image_confidence = predict(img_pil)

    answers = dict(zip(_SURVEY_FIELD_NAMES, values))
    lifestyle_prob, lifestyle_status, _details = predict_lifestyle_risk(answers)

    result = combine(
        image_prob=image_confidence,
        lifestyle_prob=lifestyle_prob,
        strategy="weighted_average",
        image_weight=image_weight,
    )

    summary = (
        f"Image pipeline: {image_status} (confidence: {image_confidence:.2f})\n"
        f"Lifestyle pipeline: {lifestyle_status} (probability: {lifestyle_prob:.2f})\n"
        f"Combined: {result['combined_risk_band']} "
        f"(fused probability: {result['combined_probability']:.2f}, "
        f"image weight: {image_weight:.2f})"
    )
    return img_with_boxes, summary


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Dental Caries Detection & Risk Prediction") as iface:
    gr.Markdown("# 🦷 Multimodal Dental Caries Detection and Risk Prediction")
    gr.Markdown(DISCLAIMER)

    with gr.Tab("Image Only"):
        gr.Markdown("Upload an intraoral dental photo. The YOLO model localizes visible carious lesions.")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Dental Image")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Detected Caries")
                image_status_output = gr.Textbox(label="Detection Status")
        image_btn = gr.Button("Run Image Detection")
        image_btn.click(fn=gradio_predict_image, inputs=image_input, outputs=[image_output, image_status_output])

    with gr.Tab("Survey Only"):
        gr.Markdown(
            "Answer the lifestyle / oral-health questions below. The model was trained on "
            "diet, brushing habits, dental visit frequency, fluoride exposure, smoking, and "
            "related behavioral/demographic factors."
        )
        if not LIFESTYLE_AVAILABLE:
            gr.Markdown(
                f"**Lifestyle model not found.** Train one first with `training/train_lifestyle_model.py`. "
                f"Details: {LIFESTYLE_ERROR}"
            )
        survey_inputs, _SURVEY_FIELD_NAMES = _build_survey_inputs()
        survey_output = gr.Textbox(label="Lifestyle-Based Risk")
        survey_btn = gr.Button("Predict Risk from Survey")
        survey_btn.click(fn=gradio_predict_survey, inputs=survey_inputs, outputs=survey_output)

    with gr.Tab("Combined (Image + Survey)"):
        gr.Markdown(
            "Combines both pipelines via late fusion to check whether image evidence and "
            "lifestyle risk factors are complementary."
        )
        combined_image_input = gr.Image(type="pil", label="Dental Image")
        image_weight_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Image pipeline weight (lifestyle gets 1 - this value)",
        )
        combined_survey_inputs, _ = _build_survey_inputs()
        with gr.Row():
            combined_image_output = gr.Image(type="pil", label="Detected Caries")
            combined_status_output = gr.Textbox(label="Combined Risk Summary", lines=4)
        combined_btn = gr.Button("Run Combined Prediction")
        combined_btn.click(
            fn=gradio_predict_combined,
            inputs=[combined_image_input, image_weight_slider] + combined_survey_inputs,
            outputs=[combined_image_output, combined_status_output],
        )

if __name__ == "__main__":
    iface.launch(debug=True)
