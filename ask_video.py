import json
import cv2
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

VIDEO_PATH = "test4.avi"
JSON_PATH = "video_data.json"

DEVICE = "cpu"


# ============================================================
# LOAD JSON
# ============================================================

print("Loading video data...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    video_data = json.load(f)


# ============================================================
# LOAD SMOLVLM
# ============================================================

print("Loading SmolVLM...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("SmolVLM loaded.")


# ============================================================
# JSON QUESTION ANSWERER
# ============================================================

def answer_from_json(question):

    q = question.lower().strip()

    objects = video_data.get("objects", [])
    summary = video_data.get("object_summary", {})

    # --------------------------------------------------------
    # Detected objects
    # --------------------------------------------------------

    if "what objects" in q or "which objects" in q:

        if not summary:
            return "No objects were detected."

        result = []

        for class_name, count in summary.items():
            result.append(
                f"{class_name}: {count}"
            )

        return "Detected objects: " + ", ".join(result)


    # --------------------------------------------------------
    # People count
    # --------------------------------------------------------

    if (
        "how many people" in q
        or "number of people" in q
        or "people count" in q
    ):

        count = summary.get("person", 0)

        return (
            f"There were {count} unique people "
            f"detected in the video."
        )


    # --------------------------------------------------------
    # Car count
    # --------------------------------------------------------

    if (
        "how many cars" in q
        or "number of cars" in q
    ):

        count = summary.get("car", 0)

        return (
            f"There were {count} unique cars "
            f"detected in the video."
        )


    # --------------------------------------------------------
    # Motorcycle count
    # --------------------------------------------------------

    if (
        "how many motorcycles" in q
        or "number of motorcycles" in q
    ):

        count = summary.get("motorcycle", 0)

        return (
            f"There were {count} unique motorcycles "
            f"detected in the video."
        )


    # --------------------------------------------------------
    # Video duration
    # --------------------------------------------------------

    if (
        "video duration" in q
        or "how long is the video" in q
        or "how long does the video" in q
    ):

        duration = video_data.get(
            "duration",
            0
        )

        return (
            f"The video duration is "
            f"{duration:.2f} seconds."
        )


    # --------------------------------------------------------
    # First person
    # --------------------------------------------------------

    if (
        "first person" in q
        and "appear" in q
    ):

        people = [
            obj for obj in objects
            if obj.get("class") == "person"
        ]

        if not people:
            return "No people were detected."

        first_person = min(
            people,
            key=lambda x: x["first_seen"]
        )

        return (
            f"The first person appeared at "
            f"{first_person['first_seen']:.2f} seconds."
        )


    # --------------------------------------------------------
    # Generic object count
    # --------------------------------------------------------

    if "how many" in q:

        for class_name, count in summary.items():

            if class_name in q:

                return (
                    f"There were {count} unique "
                    f"{class_name} objects detected."
                )


    # --------------------------------------------------------
    # No JSON answer
    # --------------------------------------------------------

    return None


# ============================================================
# EXTRACT FRAMES WITH OPENCV
# ============================================================

def extract_frames(
    video_path,
    number_of_frames=6
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            "No frames found in video."
        )

    frames = []

    for i in range(number_of_frames):

        frame_index = int(
            i * (total_frames - 1)
            / max(number_of_frames - 1, 1)
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        frames.append(image)

    cap.release()

    return frames


# ============================================================
# SMOLVLM VISUAL QUESTION
# ============================================================

def ask_smolvlm(question):

    print("Extracting frames with OpenCV...")

    images = extract_frames(
        VIDEO_PATH,
        number_of_frames=6
    )

    if not images:
        return "Could not extract video frames."


    print(
        f"Sending {len(images)} frames to SmolVLM..."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # These are IMAGE inputs, NOT VIDEO inputs.
    # This avoids TorchCodec completely.
    # --------------------------------------------------------

    content = []

    for image in images:

        content.append({
            "type": "image",
            "image": image
        })


    content.append({
        "type": "text",
        "text": question
    })


    messages = [
        {
            "role": "user",
            "content": content
        }
    ]


    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )


    # CPU
    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=100
        )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]


    return answer.strip()


# ============================================================
# MAIN
# ============================================================

print()
print("================================")
print("Video Question Answering")
print("Type 'exit' to stop")
print("================================")


while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue


    # ========================================================
    # STEP 1: TRY JSON FIRST
    # ========================================================

    answer = answer_from_json(question)


    if answer is not None:

        print()
        print("Source: video_data.json")
        print("Answer:", answer)

        continue


    # ========================================================
    # STEP 2: USE SMOLVLM
    # ========================================================

    print()
    print("Source: SmolVLM")

    try:

        answer = ask_smolvlm(question)

        print()
        print("Answer:", answer)

    except Exception as e:

        print()
        print("SmolVLM error:")
        print(e)
