import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import warnings
warnings.filterwarnings("ignore")

# Проверка доступности GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def load_and_prepare_dataset():
    """
    Загрузка и подготовка датасета bond005/sberdevices_golos_10h_crowd
    """
    print("Загрузка датасета SberDevices Golos 10h Crowd...")
    
    # Загрузка датасета
    dataset = load_dataset("bond005/sberdevices_golos_10h_crowd")
    
    # Проверка структуры датасета
    print(f"Структура датасета: {dataset}")
    print(f"Примеры колонок: {dataset['train'].column_names}")
    
    # Если датасет не разделен, создаем разделение
    if isinstance(dataset, DatasetDict):
        train_dataset = dataset["train"]
        # Создаем validation из части train (10%)
        split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    
    # Проверка примера данных
    print("\nПример данных из датасета:")
    example = dataset["train"][0]
    print(f"Аудио файл: {example['audio']['path'] if 'path' in example['audio'] else 'in memory'}")
    print(f"Длина аудио: {len(example['audio']['array'])} samples")
    print(f"Частота дискретизации: {example['audio']['sampling_rate']} Hz")
    print(f"Транскрипция: {example['sentence']}")
    
    return dataset

# Загрузка датасета
dataset = load_and_prepare_dataset()

def setup_model_and_processor(model_name="openai/whisper-small"):
    """
    Инициализация модели Whisper, токенизатора и процессора
    """
    print(f"\nЗагрузка модели {model_name}...")
    
    # Загрузка feature extractor и токенизатора
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Russian", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="Russian", task="transcribe")
    
    # Загрузка модели
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Установка языка для генерации
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="russian", 
        task="transcribe"
    )
    
    # Настройка модели для смешанной точности (если GPU доступен)
    if torch.cuda.is_available():
        model = model.to("cuda")
        model.config.use_cache = False
    
    print(f"Модель загружена: {model_name}")
    print(f"Размер модели: {sum(p.numel() for p in model.parameters()):,} параметров")
    
    return model, processor, feature_extractor, tokenizer

# Инициализация (можно выбрать другую модель: tiny, base, small, medium)
MODEL_NAME = "openai/whisper-small"  # или "openai/whisper-tiny" для быстрого тестирования
model, processor, feature_extractor, tokenizer = setup_model_and_processor(MODEL_NAME)


def prepare_dataset(batch):
    """
    Подготовка батча данных для обучения
    """
    # Ресемплинг до 16kHz, если необходимо
    audio = batch["audio"]
    
    # Извлечение признаков
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Токенизация текста
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch

# Применяем предобработку к датасету
print("\nПредобработка данных...")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Подготовка датасета
encoded_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    num_proc=4  # Количество процессов для параллельной обработки
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Разделение признаков и меток
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Пакетирование аудио признаков
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Пакетирование меток
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Замена pad token id на -100 для игнорирования в функции потерь
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Установка decoder_start_token_id для начала генерации
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch

# Инициализация data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Загрузка метрики WER (Word Error Rate)
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    """
    Вычисление метрик WER и CER
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Замена -100 на pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Декодирование предсказаний и меток
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Вычисление метрик
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-sberdevices-russian",  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=4,   
    gradient_accumulation_steps=2,  
    learning_rate=1e-5,             
    warmup_steps=500,               
    max_steps=2000,                 
    gradient_checkpointing=True,    
    fp16=torch.cuda.is_available(), 
    evaluation_strategy="steps",    
    eval_steps=200,                 
    save_strategy="steps",          
    save_steps=200,                 
    logging_strategy="steps",       
    logging_steps=50,               
    report_to="tensorboard",        
    load_best_model_at_end=True,    
    metric_for_best_model="wer",    
    greater_is_better=False,       
    push_to_hub=False,             
    seed=42,                       
    dataloader_num_workers=4,      
    predict_with_generate=True,    
    generation_max_length=225,     
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


print("\n" + "="*50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*50)

# Запуск обучения
train_result = trainer.train()

# Сохранение финальной модели
trainer.save_model()
processor.save_pretrained(training_args.output_dir)

# Сохранение логов обучения
trainer.save_state()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

print("\n" + "="*50)
print("ОЦЕНКА МОДЕЛИ")
print("="*50)

# Оценка на валидационном наборе
eval_metrics = trainer.evaluate()
print(f"\nМетрики на валидационном наборе:")
print(f"WER: {eval_metrics['eval_wer']:.4f}")
print(f"CER: {eval_metrics['eval_cer']:.4f}")

def transcribe_audio(audio_path, model, processor):
    """
    Транскрибирование аудио файла
    """
    import librosa
    
    # Загрузка аудио
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Подготовка входных данных
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    
    # Перенос на GPU если доступно
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")
    
    # Генерация транскрипции
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="russian",
            task="transcribe"
        )
    
    # Декодирование
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def save_model_for_inference(model_path, output_path):
    """
    Сохранение модели для инференса
    """
    import os
    
    # Создание директории если не существует
    os.makedirs(output_path, exist_ok=True)
    
    # Сохранение модели и токенизатора
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    print(f"\nМодель сохранена в: {output_path}")
    print("Для загрузки модели используйте:")
    print(f"model = WhisperForConditionalGeneration.from_pretrained('{output_path}')")
    print(f"processor = WhisperProcessor.from_pretrained('{output_path}')")

# Сохранение модели
save_model_for_inference(
    model_path=training_args.output_dir,
    output_path="./whisper-sberdevices-russian-final"
)

print("\n" + "="*50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("="*50)