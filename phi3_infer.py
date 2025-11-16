#!/usr/bin/env python3
import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
LORA_ADAPTER = None  # или путь к твоему адаптеру, если есть

SYSTEM_PROMPT = (
    "Твоя задача — анализировать текст и присваивать ему возрастной рейтинг.\n"
    "ПРОЦЕДУРА АНАЛИЗА:\n"
    "1. Внимательно прочитай текст.\n"
    "2. Ищи конкретные нарушения по категориям:\n"
    "   - НАСИЛИЕ: оружие, драки, убийства, кровь, жестокость.\n"
    "   - НЕНОРМАТИВНАЯ ЛЕКСИКА: мат, ругательства, оскорбления.\n"
    "   - СЕКСУАЛЬНЫЙ КОНТЕНТ: интимные сцены, обнажение, эротика.\n"
    "   - АЛКОГОЛЬ/НАРКОТИКИ: только если в тексте ЯВНО описано употребление или пропаганда "
    "конкретных веществ (например: пьёт, напился, курит, нюхает, колется, принял таблетку, "
    "под кайфом, торгует наркотиками и т.п.).\n"
    "   - ОТДЕЛЯЙ имена собственные и фамилии от веществ. Если слово похоже на наркотик, "
    "но используется как имя персонажа (например, «Белладонна Тукк»), НЕ считай это "
    "категорией АЛКОГОЛЬ/НАРКОТИКИ.\n"
    "   - ПУГАЮЩИЙ КОНТЕНТ: ужасы, психологическое давление.\n\n"
    "ВОЗРАСТНЫЕ КАТЕГОРИИ:\n"
    "0+  - Полностью безопасно, детский контент.\n"
    "6+  - Мягкие условности (персонажи в опасности без деталей).\n"
    "12+ - Умеренное насилие без крови, лёгкий испуг.\n"
    "16+ - Явное насилие, алкоголь/табак, сексуальные отсылки.\n"
    "18+ - Жестокость, откровенный секс, наркотики, тяжёлые темы.\n\n"
    "ПРАВИЛА ОТВЕТА:\n"
    "- Анализируй КОНКРЕТНО этот текст, а не шаблонно.\n"
    "- В поле \"why\" укажи реальную причину на русском.\n"
    "- В поле \"label\" укажи основную категорию нарушения.\n"
    "- Если нарушений нет — ставь 0+.\n"
    "- Будь строгим, но справедливым.\n\n"
    "- При сомнении (когда слово может быть и именем, и веществом) — выбирай более мягкий "
    "вариант и НЕ ставь категорию АЛКОГОЛЬ/НАРКОТИКИ без явного употребления.\n"
    "ФОРМАТ ОТВЕТА:\n"
    "- Выводи СТРОГО ОДИН JSON-объект.\n"
    "- Никакого дополнительного текста ДО или ПОСЛЕ JSON.\n"
    "- Используй только двойные кавычки для ключей и значений.\n"
    "Только один объект формата:\n"
    '{\"rating\": \"...\", \"why\": \"...\", \"label\": \"...\"}\n'
)



def build_prompt(message: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + "Вот текст для анализа:\n"
        + message
        + "\n\nОтветь ОДНИМ JSON-объектом:"
    )

def extract_json(text: str) -> dict:
    # Ищем ПЕРВЫЙ ненаслоенный блок {...}
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if not m:
        return {}

    snippet = m.group(0).strip()

    # Сначала пробуем как нормальный JSON с двойными кавычками
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # Если не получилось — пробуем как python-словарь с одинарными кавычками
    try:
        import ast
        data = ast.literal_eval(snippet)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
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

    msg = input("Введите предложение: ").strip()
    if not msg:
        print("Пустой ввод, выхожу.")
        return

    prompt = build_prompt(msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,  # вместо 80
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    
    # сколько токенов было на входе
    input_len = inputs["input_ids"].shape[-1]
    
    # берём только сгенерированные токены
    gen_tokens = out[0][input_len:]
    
    completion = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Обрезаем всё после первой закрывающей фигурной скобки
    end = completion.find("}")
    if end != -1:
        completion = completion[:end+1]
    
    print("ОБРЕЗАННЫЙ ОТВЕТ:", repr(completion))
    
    #data = extract_json(completion)
    #print(data)



if __name__ == "__main__":
    main()
