import evaluate
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np

def evaluate_distilgpt2_rouge(test_texts, max_length=50, device=0):
    print("Загрузка модели DistilGPT2...")
    generator = pipeline(
        "text-generation", 
        model="distilgpt2",
        device=device if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    rouge = evaluate.load("rouge")
    
    all_predictions = []
    all_references = []

    print("Начинаем оценку модели...")
    
    for text in tqdm(test_texts, desc="Оценка текстов"):
        if len(text.split()) < 8:  # Пропускаем слишком короткие тексты
            continue
            
        # Разделяем текст: 3/4 для контекста, 1/4 для проверки
        words = text.split()
        split_point = int(len(words) * 0.75)
        
        context = ' '.join(words[:split_point])
        reference = ' '.join(words[split_point:])
        
        # Генерируем продолжение
        try:
            result = generator(
                context,
                max_length=len(context.split()) + len(reference.split()) + 10,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = result[0]["generated_text"]
            
            # Извлекаем сгенерированную часть (после контекста)
            generated_continuation = generated_text[len(context):].strip()
            
            all_predictions.append(generated_continuation)
            all_references.append(reference)
            
        except Exception as e:
            print(f"Ошибка при генерации: {e}")
            continue
    
    # Вычисляем метрики ROUGE
    if all_predictions and all_references:
        rouge_results = rouge.compute(
            predictions=all_predictions,
            references=all_references,
            use_stemmer=True
        )
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ DistilGPT2")
        print("="*50)
        print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"Оценено текстов: {len(all_predictions)}")
        
        # Показываем несколько примеров
        print("\nПримеры предсказаний:")
        for i in range(min(3, len(all_predictions))):
            print(f"\nПример {i+1}:")
            print(f"Контекст: {test_texts[i].split()[:int(len(test_texts[i].split())*0.75)]}...")
            print(f"Ожидалось: {all_references[i]}")
            print(f"Предсказано: {all_predictions[i]}")
            
        return rouge_results
    else:
        print("Не удалось получить предсказания для вычисления метрик")
        return None