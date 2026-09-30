import os, json, torch, re, gc, pandas as pd
from transformers import Pipeline, pipeline, logging as hf_logging
from diffusers import StableDiffusionXLPipeline
from PIL import Image

hf_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

text_generator: Pipeline = pipeline(
   "text-generation"
   ,model="Qwen/Qwen3.5-2B"
   ,torch_dtype=torch.bfloat16
   ,device_map="auto"
)

os.makedirs("images", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

topics = ["Робототехника", "Искусственный интеллект", "Квантовые вычисления", "Биотехнологии", "Нейронные сети"]
subtopics = 60
dataset0_records = []
dataset1_records = []

print(f"Генерация текста на {device}...") # ↓↓↓

prompts = []
keys = [] 

for topic in topics:
    for sub_ids in range(subtopics):
        
        messages = [
                {"role": "system", "content": (
                "Ты пишешь тексты размером в 300 слов и промпты в 50-60 слов для Stable Diffusion. "
                "Ты возвращаешь ТОЛЬКО JSON с РОВНО двумя полями: "
                '"description" и "image_prompt". '
                "Никаких других полей. Никаких пояснений. Только JSON. "
                'Формат вывода: {"description": "text", "image_prompt": "prompt"}'
                )},
                {"role": "user", "content": (
                f"Напиши текст на тему {topic}, размером в 300 слов. " "Также напиши промпт на английском языке для генерации изображения на эту тему размером в 50-60 слов."
                )},
        ]
        prompts.append(messages)
        keys.append((topic, sub_ids))


all_results = []
batch_size = 4


for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    outputs = text_generator(
        batch
        ,max_new_tokens=2048
        ,temperature=0.5
        ,top_p=0.9
        ,do_sample=True
        ,repetition_penalty=1.1
        )
    
    for output in outputs:
        #print(output)
        all_results.append(output[0]["generated_text"][-1]["content"])

for (topic, sub_ids), output in zip(keys, all_results):
    generated_text = output.replace("'image_prompt'", '"image_prompt"')
    try: 
        data = json.loads(generated_text)
    except Exception:
        print(f"Ошибка при разборе JSON для темы '{topic}': {generated_text}")
        continue

    if "description" not in data or "image_prompt" not in data:
        print(f"[{topic}_{sub_ids}] Неполный JSON, есть поля: {list(data.keys())}")
        continue

    words = len(re.findall(r"\w+", data["description"]))
    tokens = len(text_generator.tokenizer.encode(data["description"]))

    dataset0_records.append({
        "topic_id": topic + f"_{sub_ids}",
        "description": data["description"],
        "image_prompt": data["image_prompt"],
        "tokens": tokens,
        "words": words,
    })


del text_generator
gc.collect()
torch.cuda.empty_cache()

print("Генерация изображений...") # ↓↓↓

image_generator = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo"
    ,torch_dtype=torch.float16
    ,variant="fp16"
).to(device)
image_generator.enable_attention_slicing()

image_prompts = [item["image_prompt"] for item in dataset0_records]
image_keys = [item["topic_id"] for item in dataset0_records]

images = []
batch_size = 2

for i in range(0, len(image_prompts), batch_size):
    batch_prompts = image_prompts[i:i+batch_size]
    with torch.no_grad():
        imgs = image_generator(batch_prompts, num_inference_steps=2, guidance_scale=0.0, height=512, width=512).images
    images.extend(imgs)

for topic_id, img, prompt in zip(image_keys, images, image_prompts):
    image = img.resize((1536, 1536), Image.LANCZOS)
    image_path = f"images/{topic_id}.png"
    image.save(image_path, format="PNG", compress_level=0)
    dataset1_records.append({
            "topic_id": topic_id,
            "image_path": image_path,
            "image_prompt": prompt,
        })

print("Сохранение датасетов...") # ↓↓↓

df0 = pd.DataFrame(dataset0_records, columns=["topic_id", "description", "image_prompt", "tokens", "words"])
df0.to_csv("datasets/texts.csv", index=False, encoding="utf-8-sig")
df1 = pd.DataFrame(dataset1_records, columns=["topic_id", "image_path", "image_prompt"])
df1.to_csv("datasets/images.csv", index=False, encoding="utf-8-sig")
print("Генерация завершена. Датасеты: datasets/texts.csv, datasets/images.csv")
