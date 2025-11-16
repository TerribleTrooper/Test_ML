#!/usr/bin/env python3
import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
LORA_ADAPTER = None  # или путь к твоему адаптеру, если есть

SYSTEM_PROMPT = (
    "ПРОЦЕДУРА АНАЛИЗА:\n"
    "1. Внимательно прочитай текст\n"
    "2. Ищи конкретные нарушения по категориям:\n"
    "   - НАСИЛИЕ: оружие, драки, убийства, кровь, жестокость\n"
    "   - НЕНОРМАТИВНАЯ ЛЕКСИКА: мат, ругательства, оскорбления\n"
    "   - СЕКСУАЛЬНЫЙ КОНТЕНТ: интимные сцены, обнажение, эротика\n"
    "   - АЛКОГОЛЬ/НАРКОТИКИ: употребление, пропаганда\n"
    "   - ПУГАЮЩИЙ КОНТЕНТ: ужасы, психологическое давление\n\n"
    "ВОЗРАСТНЫЕ КАТЕГОРИИ:\n"
    "0+  - Полностью безопасно, детский контент\n"
    "6+  - Мягкие условности (персонажи в опасности без деталей)\n"
    "12+ - Умеренное насилие без крови, легкий испуг\n"
    "16+ - Явное насилие, алкоголь/табак, сексуальные отсылки\n"
    "18+ - Жестокость, откровенный секс, наркотики, тяжелые темы\n\n"
    "ПРАВИЛА ОТВЕТА:\n"
    "- Анализируй КОНКРЕТНО этот текст, а не шаблонно\n"
    "- В поле 'why' укажи реальную причину на русском\n"
    "- В поле 'label' укажи основную категорию нарушения\n"
    "- Если нарушений нет - ставь 0+\n"
    "- Будь строгим но справедливым\n\n"
    ФОРМАТ ОТВЕТА:
    "- Выводи СТРОГО ОДИН JSON-объект./n"
    "- Никакого дополнительного текста ДО или ПОСЛЕ JSON./n"
    "- Не пиши "ТЕКСТ ДЛЯ АНАЛИЗА" в ответе./n"
    "- Не пиши примеры./n"
    "Только один объект формата:/n"
    "{"rating": "...", "why": "...", "label": "..."}/n"

    "ПРИМЕРЫ РАЗБОРА:\n"
    "Текст: 'Котенок играл с мячиком' → "
    "{'rating':'0+','why':'безопасный детский контент','label':'без нарушений'}\n"
    "Текст: 'Он ударил его кулаком по лицу' → "
    "{'rating':'16+','why':'сцена физического насилия','label':'насилие'}\n"
    "Текст: 'Она выругалась матом' → "
    "{'rating':'18+','why':'ненормативная лексика','label':'нецензурная лексика'}\n"
    "Текст: 'Они выпили вина за ужином' → "
    "{'rating':'16+','why':'упоминание алкоголя','label':'алкоголь'}\n"
    "Текст: 'В темноте послышался страшный шорох' → "
    "{'rating':'12+','why':'элементы страха и напряжения','label':'страшное'}\n"
)


def build_prompt(message: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + f"ТЕКСТ ДЛЯ АНАЛИЗА: {message}\n\n"
        + "ТВОЙ АНАЛИЗ (ТОЛЬКО JSON):"
    )

def extract_json(text: str) -> dict:
    # Ищем первый блок {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

def main():
    # Принудительно используем CPU, чтобы не ловить OOM на MPS
    device = "cpu"
    print(f"Использую устройство: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # норм для CPU
        device_map=None,
    ).to(device)

    if LORA_ADAPTER:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER).to(device)

    while True:
        msg = input("Введите предложение (пусто — выход): ").strip()
        if not msg:
            break

        prompt = build_prompt(msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = decoded[len(prompt):].strip()
        print("СЫРОЙ ОТВЕТ МОДЕЛИ:\n", completion)
        print()

        data = extract_json(completion)
        print("РАЗОБРАННЫЙ JSON:", data)
        print("-" * 40)

if __name__ == "__main__":
    main()
