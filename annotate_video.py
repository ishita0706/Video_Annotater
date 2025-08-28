import streamlit as st
import torch
import tempfile
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import supervision as sv
import os

st.title("Video Annotator")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "vikhyatk/moondream2"
revision = "2025-06-21"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    return tokenizer, model

tokenizer, model = load_model()

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv"])
prompt_list = st.text_input("Object prompts (comma-separated, e.g., person, car)", "")

if "highlight_list" not in st.session_state:
    st.session_state.highlight_list = []
check=True
if st.checkbox("Add red highlight intervals"):
    with st.form(key="highlight_form"):
        start = st.number_input("Start Time (sec)", 0.0, format="%.2f", key="start")
        end = st.number_input("End Time (sec)", 1.0, format="%.2f", key="end")
        object_name = st.text_input("Object to be highlight red(e.g., person)", key="obj")
        color_label=st.text_input("Label of the red highlighted box",key="lb")
        ids = st.text_input("IDs of this object to highlight in RED (comma-separated)", "0", key="ids")
        add_config = st.form_submit_button("Add Interval")

        if add_config:
            ids_list = [int(i.strip()) for i in ids.split(",") if i.strip().isdigit()]
            st.session_state.highlight_list.append((start, end, object_name.strip().lower(), color_label,ids_list))
            st.success(f"Added: {object_name} ({ids_list}) from {start:.2f}s to {end:.2f}s")
else:
    check=False
if st.session_state.highlight_list:
    st.write("### Current Highlight Intervals:")
    for idx, (start, end, obj, lb,ids) in enumerate(st.session_state.highlight_list):
        st.write(f"{idx+1}. {start:.2f}s - {end:.2f}s | Object: '{obj}' | IDs: {ids} | Label:{lb}")

highlight_config = st.session_state.highlight_list

def get_color(track_id, t, label):
    for start, end, obj_name,lb, ids in highlight_config:
        if start <= t <= end and track_id in ids and obj_name.lower() == label.lower():
            return sv.Color.RED,lb
    return sv.Color.GREEN,""

if st.button("Run Annotation") and video_file:
    with st.spinner("Processing video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(video_file.read())
            input_path = temp_input.name
        output_path = input_path.replace(".mp4", "_annotated.mp4")
        object_prompts = [x.strip() for x in prompt_list.split(",")]

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        tracker = sv.ByteTrack()

        frame_idx = 0
        progress_bar = st.progress(0, text="Processing video frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            boxes, labels, class_ids, confidences = [], [], [], []
            for class_id, prompt in enumerate(object_prompts):
                try:
                    if frame_idx % 7 == 1:
                        results = model.detect(pil_image, prompt)["objects"]
                    for obj in results:
                        x1 = obj["x_min"] * width
                        y1 = obj["y_min"] * height
                        x2 = obj["x_max"] * width
                        y2 = obj["y_max"] * height
                        boxes.append([x1, y1, x2, y2])
                        labels.append(prompt)
                        class_ids.append(class_id)
                        confidences.append(obj.get("confidence", 1.0))
                except Exception as e:
                    st.warning(f"Detection failed on frame {frame_idx}: {e}")

            detections = sv.Detections(
                xyxy=np.array(boxes) if boxes else np.empty((0, 4)),
                class_id=np.array(class_ids) if class_ids else np.array([]),
                confidence=np.array(confidences) if confidences else np.array([]),
                data={"label": np.array(labels) if labels else np.array([])}
            )

            detections = tracker.update_with_detections(detections)
            annotated_labels = [
                f"{label} (ID: {tracker_id})"
                for label, tracker_id in zip(
                    detections.data.get("label", []),
                    detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)
                )
            ]
            
            labels = detections.data.get("label", [])
            colors_labels = [get_color(tid, timestamp, label) for tid, label in zip(detections.tracker_id, labels)]

            for box, label, (color, lb), tid in zip(detections.xyxy, labels, colors_labels, detections.tracker_id):
                x1, y1, x2, y2 = box.astype(int)
                if check:
                    label = lb  
                else:
                    label=f"ID: {tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.as_bgr(), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.as_bgr(), 4)

            out.write(frame)
            progress_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Processed {frame_idx}/{total_frames} frames")


        cap.release()
        out.release()

    progress_bar.empty()
    st.success("Video processing complete")
    st.session_state.clear()
    with open(output_path, "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="processed_video.mp4")

    
