import evaluate
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

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
    

def generate_text(model, initial_tokens, vocab, device, max_length=50, temperature=1.0, top_k=5):
    model.eval()
    generated_tokens = initial_tokens.copy()
    
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    print(f"\nНачальный контекст: {' '.join(initial_tokens)}")
    print("ГЕНЕРАЦИЯ ТЕКСТА:")
    
    with torch.no_grad():
        for step in range(max_length):
            # Подготавливаем входные данные
            input_ids = []
            for token in generated_tokens:
                input_ids.append(vocab.get(token, vocab['<UNK>']))
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # Получаем предсказание
            output, _ = model(input_tensor)
            output = output / temperature
            
            # Применяем top-k sampling
            probs = F.softmax(output, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # Выбираем случайный токен из top-k
            chosen_idx = torch.multinomial(top_probs[0], 1).item()
            next_token_idx = top_indices[0][chosen_idx].item()
            next_token = reverse_vocab.get(next_token_idx, '<UNK>')
            
            # Добавляем к сгенерированному тексту
            generated_tokens.append(next_token)
            
            # Останавливаемся на точке или подобном знаке
            if next_token in ['.', '!', '?'] and step > 10:
                break
    
    generated_text = ' '.join(generated_tokens)
    return generated_text

def calculate_similarity(str1, str2):
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    return len(intersection) / max(len(words1), len(words2))

def evaluate_text_generation(model, test_cases, vocab, device):
    print("\n" + "ОЦЕНКА КАЧЕСТВА ГЕНЕРАЦИИ" + "\n")

    for i, (initial_tokens, expected_continuation) in enumerate(test_cases, 1):
        generated_text = generate_text(
            model, 
            initial_tokens, 
            vocab, 
            device,
            max_length=30,
            temperature=0.7
        )
        
        generated_continuation = ' '.join(generated_text.split()[len(initial_tokens):])
        
        print(f"\nТест {i}:")
        print(f"Контекст:    {' '.join(initial_tokens)}")
        print(f"Ожидалось:   {expected_continuation}")
        print(f"Сгенерировано: {generated_continuation}")
        print(f"Схожесть:    {calculate_similarity(expected_continuation, generated_continuation):.2f}")

def compare_models_generation(lstm_model, test_cases, vocab, device, gpt2_generator, gpt2_tokenizer):
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ГЕНЕРАЦИИ LSTM vs DistilGPT2")
    print("="*60)
    
    for i, (initial_tokens, expected_continuation) in enumerate(test_cases, 1):
        # Генерация LSTM
        lstm_generated = generate_text(
            lstm_model, 
            initial_tokens, 
            vocab, 
            device,
            max_length=30,
            temperature=0.7
        )
        lstm_continuation = ' '.join(lstm_generated.split()[len(initial_tokens):])
        
        # Генерация DistilGPT2
        context = ' '.join(initial_tokens)
        try:
            gpt2_result = gpt2_generator(
                context,
                max_length=len(context.split()) + 15,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=gpt2_tokenizer.eos_token_id
            )
            gpt2_full_text = gpt2_result[0]["generated_text"]
            gpt2_continuation = gpt2_full_text[len(context):].strip()
        except Exception as e:
            print(f"Ошибка GPT2 генерации: {e}")
            gpt2_continuation = "Ошибка генерации"
        
        # Вычисление схожести
        lstm_similarity = calculate_similarity(expected_continuation, lstm_continuation)
        gpt2_similarity = calculate_similarity(expected_continuation, gpt2_continuation)
        
        print(f"\nТЕСТ {i}:")
        print(f"Контекст:     {context}")
        print(f"Ожидалось:    {expected_continuation}")
        print(f"LSTM:         {lstm_continuation} (схожесть: {lstm_similarity:.2f})")
        print(f"DistilGPT2:   {gpt2_continuation} (схожесть: {gpt2_similarity:.2f})")
        print("-" * 80)