import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import (
    MBartTokenizer,
    MBartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import random
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Скачивание необходимых ресурсов для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Проверка доступности GPU
print(f"Дата и время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Используется CPU для обучения")

# Установка seed для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_and_analyze_gazeta_dataset():
    """
    Загрузка и анализ датасета Gazeta для суммаризации
    """
    print("\n" + "="*60)
    print("ЗАГРУЗКА ДАТАСЕТА GAZETA")
    print("="*60)
    
    # Загрузка датасета Gazeta
    print("Загрузка датасета Gazeta...")
    dataset = load_dataset("IlyaGusev/gazeta")
    
    # Проверка структуры датасета
    print(f"\nСтруктура датасета:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} примеров")
    
    # Проверка доступных колонок
    print(f"\nКолонки в train split:")
    for col in dataset["train"].column_names:
        print(f"  - {col}")
    
    # Анализ примера данных
    print("\nАнализ примера данных:")
    example = dataset["train"][0]
    print(f"Тип текста: {type(example['text'])}")
    print(f"Тип суммаризации: {type(example['summary'])}")
    
    # Вычисление статистики по длине текстов
    print("\nСтатистика по длине текстов (первые 1000 примеров):")
    train_sample = dataset["train"].select(range(min(1000, len(dataset["train"]))))
    
    text_lengths = [len(text.split()) for text in train_sample["text"]]
    summary_lengths = [len(summary.split()) for summary in train_sample["summary"]]
    
    print(f"Средняя длина текста: {np.mean(text_lengths):.1f} слов")
    print(f"Медиана длины текста: {np.median(text_lengths):.1f} слов")
    print(f"Средняя длина суммаризации: {np.mean(summary_lengths):.1f} слов")
    print(f"Медиана длины суммаризации: {np.median(summary_lengths):.1f} слов")
    print(f"Соотношение длина_суммаризации/длина_текста: {(np.mean(summary_lengths)/np.mean(text_lengths)*100):.1f}%")
    
    # Показ примера
    print("\n" + "="*60)
    print("ПРИМЕР ДАННЫХ ИЗ ДАТАСЕТА:")
    print("="*60)
    print(f"\nИсходный текст (первые 500 символов):")
    print(example['text'][:500] + "...")
    print(f"\nСуммаризация:")
    print(example['summary'])
    print(f"\nКатегория: {example.get('title', 'Не указана')}")
    
    return dataset

# Загрузка датасета
dataset = load_and_analyze_gazeta_dataset()

def prepare_and_split_data(dataset, test_size=0.1, val_size=0.1):
    """
    Подготовка и разделение данных на train/validation/test
    """
    print("\n" + "="*60)
    print("ПОДГОТОВКА И РАЗДЕЛЕНИЕ ДАННЫХ")
    print("="*60)
    
    # Берем только train split из оригинального датасета
    train_data = dataset["train"]
    
    # Разделяем на train, validation и test
    print(f"Исходный размер train split: {len(train_data)} примеров")
    
    # Сначала отделяем test set
    train_test_split = train_data.train_test_split(
        test_size=test_size, 
        seed=SEED
    )
    
    # Затем разделяем оставшиеся данные на train и validation
    train_val_split = train_test_split["train"].train_test_split(
        test_size=val_size/(1-test_size), 
        seed=SEED
    )
    
    # Создаем финальный DatasetDict
    final_dataset = DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    })
    
    print(f"\nФинальное распределение данных:")
    print(f"  Train: {len(final_dataset['train'])} примеров")
    print(f"  Validation: {len(final_dataset['validation'])} примеров")
    print(f"  Test: {len(final_dataset['test'])} примеров")
    
    return final_dataset

# Подготовка данных
dataset = prepare_and_split_data(dataset)


def setup_mbart_model():
    """
    Инициализация модели MBart для суммаризации на русском языке
    """
    print("\n" + "="*60)
    print("ЗАГРУЗКА МОДЕЛИ MBART")
    print("="*60)
    
    model_name = "IlyaGusev/mbart_ru_sum_gazeta"
    print(f"Загрузка предобученной модели: {model_name}")
    
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = MBartTokenizer.from_pretrained(
        model_name,
        src_lang="ru_RU",
        tgt_lang="ru_RU"
    )
    
    # Загрузка модели
    print("Загрузка модели...")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    # Перемещение модели на GPU если доступно
    model = model.to(device)
    
    # Информация о модели
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nИнформация о модели:")
    print(f"  Общее количество параметров: {num_params:,}")
    print(f"  Обучаемые параметры: {trainable_params:,}")
    print(f"  Размер словаря: {model.config.vocab_size:,}")
    print(f"  Максимальная длина позиционных эмбеддингов: {model.config.max_position_embeddings}")
    print(f"  Количество слоев энкодера: {model.config.encoder_layers}")
    print(f"  Количество слоев декодера: {model.config.decoder_layers}")
    print(f"  Размер скрытого состояния: {model.config.d_model}")
    
    return model, tokenizer

# Инициализация модели
model, tokenizer = setup_mbart_model()


def preprocess_function(examples, max_input_length=1024, max_target_length=128):
    """
    Функция предобработки данных для суммаризации
    """
    # Токенизация входных текстов
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    
    # Токенизация целевых суммаризаций
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
    
    # Заменяем pad_token_id на -100 для игнорирования в функции потерь
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

print("\n" + "="*60)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("="*60)

# Конфигурация длин
MAX_INPUT_LENGTH = 1024  # Максимальная длина входного текста
MAX_TARGET_LENGTH = 128  # Максимальная длина суммаризации

print(f"Максимальная длина входного текста: {MAX_INPUT_LENGTH} токенов")
print(f"Максимальная длина суммаризации: {MAX_TARGET_LENGTH} токенов")

# Применяем предобработку ко всем сплитам
tokenized_datasets = dataset.map(
    lambda x: preprocess_function(x, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Токенизация данных"
)

print("\nПример токенизированных данных:")
print(f"Размер train после токенизации: {len(tokenized_datasets['train'])}")
print(f"Размер validation после токенизации: {len(tokenized_datasets['validation'])}")

# Проверка примера токенизированных данных
sample = tokenized_datasets["train"][0]
print(f"\nРазмеры в примере:")
print(f"  input_ids: {len(sample['input_ids'])} токенов")
print(f"  attention_mask: {len(sample['attention_mask'])}")
print(f"  labels: {len(sample['labels'])} токенов")

# Декодируем обратно для проверки
print(f"\nДекодированный текст (первые 100 токенов):")
print(tokenizer.decode(sample['input_ids'][:100], skip_special_tokens=True))
print(f"\nДекодированная суммаризация:")
labels = [token for token in sample['labels'] if token != -100]
print(tokenizer.decode(labels, skip_special_tokens=True))

print("\n" + "="*60)
print("НАСТРОЙКА DATA COLLATOR")
print("="*60)

# Создаем DataCollator для динамического padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    max_length=MAX_INPUT_LENGTH,
    label_pad_token_id=-100,
)

print("DataCollator настроен для динамического padding батчей")

def compute_metrics(eval_pred):
    """
    Вычисление метрик ROUGE для оценки качества суммаризации
    """
    predictions, labels = eval_pred
    
    # Декодируем предсказания
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Заменяем -100 в метках на pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Вычисляем ROUGE метрики
    rouge = evaluate.load("rouge")
    
    # Вычисляем ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True
    )
    
    # Извлекаем основные метрики
    result = {key: value * 100 for key, value in result.items()}
    
    # Добавляем среднюю длину предсказаний
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

print("\n" + "="*60)
print("НАСТРОЙКА МЕТРИК ОЦЕНКИ")
print("="*60)
print("Будут использоваться метрики ROUGE:")
print("  - ROUGE-1: совпадение униграмм")
print("  - ROUGE-2: совпадение биграмм")
print("  - ROUGE-L: совпадение на основе наибольшей общей подпоследовательности")

print("\n" + "="*60)
print("НАСТРОЙКА АРГУМЕНТОВ ОБУЧЕНИЯ")
print("="*60)

# Определяем имя модели для сохранения
model_name = "mbart_ru_sum_gazeta_finetuned"
output_dir = f"./{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    
    # Параметры обучения
    num_train_epochs=3,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8, 
    
    # Оптимизатор
    learning_rate=5e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    
    # Планировщик
    lr_scheduler_type="linear",
    warmup_steps=500,
    
    # Смешанная точность
    fp16=torch.cuda.is_available(),
    
    # Экономия памяти
    gradient_checkpointing=True,
    
    # Стратегии
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    
    # Генерация
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,
    
    # Разное
    report_to="tensorboard",
    seed=SEED,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    push_to_hub=False,
)

print("Аргументы обучения установлены:")
print(f"  Выходная директория: {training_args.output_dir}")
print(f"  Количество эпох: {training_args.num_train_epochs}")
print(f"  Размер батча: {training_args.per_device_train_batch_size}")
print(f"  Накопление градиентов: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Смешанная точность (fp16): {training_args.fp16}")
print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")


print("\n" + "="*60)
print("ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА")
print("="*60)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# Запуск обучения
train_result = trainer.train()

print(f"\nОбучение завершено! Время окончания: {datetime.now().strftime('%H:%M:%S')}")

# Сохранение результатов обучения
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)

# Сохранение метрик обучения
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

print(f"\nМодель сохранена в: {training_args.output_dir}")


print("\n" + "="*60)
print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОМ НАБОРЕ")
print("="*60)

# Оценка на тестовом наборе
test_results = trainer.evaluate(
    eval_dataset=tokenized_datasets["test"],
    metric_key_prefix="test"
)

print("\nРезультаты на тестовом наборе:")
for key, value in test_results.items():
    if "test_" in key:
        print(f"  {key}: {value:.4f}")

def generate_summary(text, model, tokenizer, max_length=128, num_beams=4):
    """
    Генерация суммаризации для заданного текста
    """
    # Токенизация входного текста
    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
        padding=True
    )
    
    # Перемещение на GPU если доступно
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Генерация суммаризации
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    
    # Декодирование
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Выбираем несколько примеров для тестирования
test_examples = dataset["test"].select(range(3))

for i, example in enumerate(test_examples):
    print(f"\n{'='*40}")
    print(f"Пример {i+1}")
    print(f"{'='*40}")
    
    text = example["text"]
    reference_summary = example["summary"]
    
    print(f"\nИсходный текст (первые 300 символов):")
    print(text[:300] + "...")
    
    print(f"\nЭталонная суммаризация:")
    print(reference_summary)
    
    # Генерация суммаризации
    generated_summary = generate_summary(text, model, tokenizer)
    
    print(f"\nСгенерированная суммаризация:")
    print(generated_summary)
    
    print(f"\nДлина исходного текста: {len(text.split())} слов")
    print(f"Длина эталонной суммаризации: {len(reference_summary.split())} слов")
    print(f"Длина сгенерированной суммаризации: {len(generated_summary.split())} слов")

# =========== 13. СОХРАНЕНИЕ ДЛЯ ИНФЕРЕНСА ===========
def save_model_for_inference(model_path, output_path):
    """
    Сохранение модели для последующего использования
    """
    import os
    import shutil
    
    # Создание директории
    os.makedirs(output_path, exist_ok=True)
    
    # Сохранение модели и токенизатора
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Создание файла конфигурации
    config = {
        "model_type": "mbart",
        "task": "summarization",
        "language": "russian",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    import json
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\nМодель сохранена для инференса в: {output_path}")
    print("Для загрузки модели используйте:")
    print(f"model = MBartForConditionalGeneration.from_pretrained('{output_path}')")
    print(f"tokenizer = MBartTokenizer.from_pretrained('{output_path}')")

# Сохранение финальной модели
final_output_path = "./mbart_ru_sum_gazeta_final"
save_model_for_inference(training_args.output_dir, final_output_path)


# Загрузка метрик обучения
import json
metrics_path = os.path.join(training_args.output_dir, "train_results.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    print("\nКлючевые метрики обучения:")
    print(f"  Общее время обучения: {metrics.get('train_runtime', 0):.1f} секунд")
    print(f"  Количество шагов: {metrics.get('train_steps', 0)}")
    print(f"  Потери на обучении: {metrics.get('train_loss', 0):.4f}")
    
    # Поиск лучших метрик валидации
    for key in metrics:
        if "eval_rouge" in key and "best" not in key:
            print(f"  {key}: {metrics[key]:.4f}")