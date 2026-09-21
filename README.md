# Video Object Tracking & Question Answering

A computer vision project that combines **YOLO11**, **Deep SORT**, and **SmolVLM2** for video analysis.

## Features

* Object detection using YOLO11
* Object tracking using Deep SORT
* Generates `video_data.json` with detected objects and timestamps
* Counts objects such as people, cars, motorcycles, etc.
* Uses SmolVLM2 for visual question answering
* Interactive command-line video questions

## Workflow

```text
Video
  ↓
YOLO11
  ↓
Deep SORT
  ↓
video_data.json
  ↓
Question
  ↓
JSON Answer / SmolVLM
```

## Installation

```bash
pip install ultralytics deep-sort-realtime opencv-python torch transformers pillow
```

## Run

First run the object tracking script:

```bash
python video_tracking.py
```

Then run the question-answering script:

```bash
python video_qa.py
```

## Example Questions

```text
What objects are in the video?
How many people are there?
How many cars are there?
How many motorcycles are there?
How long is the video?
When did the first person appear?
```

## Output

The project generates:

```text
video_data.json
```

It also displays the processed video with bounding boxes and tracking IDs.

## Models

* YOLO11
* Deep SORT
* Hugging Face SmolVLM2

## Demo

Add the output video/GIF here:

```text
assets/demo.gif
```
