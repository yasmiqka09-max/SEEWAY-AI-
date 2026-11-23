import os
import time
import gradio as gr
from ultralytics import YOLO

print("Загрузка SEEWAY AI...")

model = YOLO("yolo11n.pt")

translations = {
    "person": "человек", "bicycle": "велосипед",
    "car": "машина", "motorcycle": "мотоцикл",
    "airplane": "самолёт", "bus": "автобус",
    "train": "поезд", "truck": "грузовик",
    "boat": "лодка", "bird": "птица",
    "cat": "кошка", "dog": "собака",
    "horse": "лошадь", "sheep": "овца",
    "cow": "корова", "elephant": "слон",
    "bear": "медведь", "zebra": "зебра",
    "giraffe": "жираф", "backpack": "рюкзак",
    "umbrella": "зонт", "handbag": "сумка",
    "suitcase": "чемодан", "bottle": "бутылка",
    "wine glass": "бокал", "cup": "чашка",
    "fork": "вилка", "knife": "нож",
    "spoon": "ложка", "bowl": "миска",
    "banana": "банан", "apple": "яблоко",
    "sandwich": "сэндвич", "orange": "апельсин",
    "broccoli": "брокколи", "carrot": "морковь",
    "pizza": "пицца", "donut": "пончик",
    "cake": "торт", "chair": "стул",
    "couch": "диван", "bed": "кровать",
    "dining table": "стол", "toilet": "унитаз",
    "tv": "телевизор", "laptop": "ноутбук",
    "mouse": "мышь", "remote": "пульт",
    "keyboard": "клавиатура", "cell phone": "телефон",
    "microwave": "микроволновка", "oven": "духовка",
    "toaster": "тостер", "sink": "раковина",
    "refrigerator": "холодильник", "book": "книга",
    "clock": "часы", "vase": "ваза",
    "scissors": "ножницы", "teddy bear": "плюшевый мишка",
    "hair drier": "фен", "toothbrush": "зубная щётка"
}


def analyze_camera(image, state):

    if image is None:
        return None, "Ожидание камеры...", "", state

    results = model(image, conf=0.45, verbose=False)
    detected_image = results[0].plot()

    current_objects = {}
    width = image.width

    for item in results[0].boxes:
        confidence = float(item.conf[0])

        if confidence < 0.45:
            continue

        class_id = int(item.cls[0])
        english_name = model.names[class_id]
        name = translations.get(english_name, english_name)

        box = item.xyxy[0].tolist()
        center_x = (box[0] + box[2]) / 2

        if center_x < width / 3:
            position = "слева"
        elif center_x > width * 2 / 3:
            position = "справа"
        else:
            position = "перед вами"

        current_objects[name] = position

    for name in list(state["seen"]):
        if name not in current_objects:
            state["seen"][name] = 0

    for name in current_objects:
        state["seen"][name] = state["seen"].get(name, 0) + 1

    for name in list(state["seen"]):
        if state["seen"][name] == 0:
            state["missing"][name] = state["missing"].get(name, 0) + 1

            if state["missing"][name] >= 5:
                state["seen"].pop(name, None)
                state["announced"].discard(name)
                state["missing"].pop(name, None)
        else:
            state["missing"].pop(name, None)

    new_objects = []

    for name, position in current_objects.items():
        if state["seen"][name] >= 2 and name not in state["announced"]:
            new_objects.append((name, position))
            state["announced"].add(name)

    if current_objects:
        result = "👁️ SEEWAY AI обнаружил:\n\n"
        for name, position in current_objects.items():
            result += f"• {name} — {position}\n"
    else:
        result = "Пока ничего не обнаружено."

    speech = ""

    if new_objects:
        phrases = [
            f"{position}: {name}"
            for name, position in new_objects
        ]
        speech = ". ".join(phrases) + "."

        # Уникальный номер заставляет Gradio замечать
        # даже повторяющиеся голосовые фразы.
        speech += "|||" + str(time.time_ns())

    return detected_image, result, speech, state


voice_js = """
(value) => {
    if (!value) return;

    const text = value.split("|||")[0].trim();
    if (!text) return;

    if (!window.speechSynthesis) {
        console.error("Браузер не поддерживает голос.");
        return;
    }

    if (!window.seewayVoiceEnabled) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ru-RU";
    utterance.rate = 1;

    const voices = window.speechSynthesis.getVoices();
    const russianVoice = voices.find(
        v => v.lang.toLowerCase().startsWith("ru")
    );

    if (russianVoice) {
        utterance.voice = russianVoice;
    }

    window.speechSynthesis.speak(utterance);
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SEEWAY AI"
) as demo:

    gr.Markdown("""
    # 👁️ SEEWAY AI
    ### Твой умный голосовой помощник

    Сначала включи русский голос, затем камеру.
    """)

    voice_button = gr.Button("🔊 Включить русский голос")
    voice_status = gr.Markdown("Голос пока не включён.")

    state = gr.State({
        "seen": {},
        "missing": {},
        "announced": set()
    })

    camera = gr.Image(
        sources=["webcam"],
        type="pil",
        streaming=True,
        label="📷 Камера"
    )

    output_image = gr.Image(label="Распознавание предметов")
    result_text = gr.Textbox(label="Что я вижу?", lines=8)
    speech_text = gr.Textbox(visible=False)

    voice_button.click(
        fn=None,
        inputs=None,
        outputs=voice_status,
        js="""
        () => {
            window.seewayVoiceEnabled = true;
            window.speechSynthesis.cancel();

            const test = new SpeechSynthesisUtterance(
                "Голосовой помощник включён."
            );

            test.lang = "ru-RU";
            window.speechSynthesis.speak(test);

            return "✅ Русский голос включён.";
        }
        """
    )

    camera.stream(
        fn=analyze_camera,
        inputs=[camera, state],
        outputs=[output_image, result_text, speech_text, state],
        stream_every=1.0,
        time_limit=300
    )

    speech_text.change(
        fn=None,
        inputs=[speech_text],
        outputs=None,
        js=voice_js
    )


demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
