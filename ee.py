import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# настройки страницы
st.set_page_config(
    page_icon = '🎬',
    layout= 'wide'
)

# заголовок
st.title('Распознавание информации по видео с датафестов и докладов ODS')
st.markdown('---')
# вкладки
tab1, tab2 = st.tabs(['Обработка аудио 🎤', 'Обработка видео 🎥'])

# кладка видео
with tab2:
    st.header('Обработка видео')
    st.info('Функционал обработки видео будет реализован в будущем')
    st.image('https://img.icons8.com/?size=100&id=10343&format=png&color=000000', width=200)

# Вкладка аудио
with tab1:
    st.header('🎤 Обработка аудиофайлов')
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        'Загрузите аудиофайл в формате WAV', 
        type=['wav'],
        help='Поддерживаются WAV файлы с русской речью'
    )
    
    if uploaded_file is not None:
        # cохраняем временный файл
        with open('temp_audio.wav', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
            
            # Показать информацию о файле
            audio, sr = librosa.load('temp_audio.wav', sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            st.info(f'📊 Длительность: {duration:.2f} сек')
        
        with col2:
            if st.button('🚀 Обработать аудио', type='primary', use_container_width=True):
                with st.spinner('Загрузка модели...'):
                    try:

                        # распознавание речи с помощью Whisper
                        @st.cache_resource
                        def load_whisper_model():
                            try:
                                processor = WhisperProcessor.from_pretrained('openai/whisper-small')
                                model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
                                
                                # Ключевая настройка: заставляем модель транскрибировать на русском
                                forced_decoder_ids = processor.get_decoder_prompt_ids(
                                    language = 'russian',
                                    task = 'transcribe'
                                )
                                
                                return processor, model, forced_decoder_ids
                            except Exception as e:
                                st.warning(f'Не удалось загрузить Whisper: {e}')
                                return None, None, None
                        
                        # загружаем модель
                        processor, model, forced_decoder_ids = load_whisper_model()
                        
                        if processor is None or model is None:
                            st.error('Не удалось загрузить модель Whisper')

                        
                        with st.spinner(f'Распознаю речь (это может занять некоторое время)...'):
                            # читаем аудио
                            audio_data, sample_rate = sf.read('temp_audio.wav')
                            
                            # конвертируем в mono если нужно
                            if audio_data.ndim > 1:
                                audio_data = audio_data.mean(axis=1)
                            
                            # ресемплируем до 16kHz (Whisper требует 16kHz)
                            if sample_rate != 16000:
                                audio_data = librosa.resample(
                                    audio_data, 
                                    orig_sr=sample_rate, 
                                    target_sr=16000
                                )
                            
                            # длительность в секундах
                            duration_seconds = len(audio_data) / 16000
                            
                            # разбиваем на части по 30 секунд
                            chunk_duration = 30
                            chunk_size = chunk_duration * 16000
                            
                            transcriptions = []
                            progress_bar = st.progress(0)
                            
                            # если файл короткий (меньше 60 секунд), обрабатываем целиком
                            if duration_seconds <= 60:
                                try:
                                    input_features = processor(
                                        audio_data, 
                                        sampling_rate = 16000, 
                                        return_tensors = 'pt'
                                    ).input_features
                                    
                                    predicted_ids = model.generate(
                                        input_features,
                                        forced_decoder_ids=forced_decoder_ids,
                                        max_new_tokens=448
                                    )
                                    
                                    transcription = processor.batch_decode(
                                        predicted_ids, 
                                        skip_special_tokens=True
                                    )[0]
                                    transcriptions.append(transcription)
                                    
                                except Exception as e:
                                    st.error(f'Ошибка при обработке короткого файла: {e}')
                            else:
                                # для длинных файлов разбиваем на части
                                num_chunks = int(np.ceil(len(audio_data) / chunk_size))
                                
                                for i in range(num_chunks):
                                    start_sample = i * chunk_size
                                    end_sample = min((i + 1) * chunk_size, len(audio_data))
                                    chunk = audio_data[start_sample:end_sample]
                                    
                                    # пропускаем тихие части
                                    if np.max(np.abs(chunk)) < 0.01:
                                        continue
                                    
                                    # обновляем прогресс-бар
                                    progress = (i + 1) / num_chunks
                                    progress_bar.progress(progress)
                                    
                                    try:
                                        input_features = processor(
                                            chunk, 
                                            sampling_rate=16000, 
                                            return_tensors='pt'
                                        ).input_features
                                        
                                        predicted_ids = model.generate(
                                            input_features,
                                            forced_decoder_ids=forced_decoder_ids,
                                            max_new_tokens=448
                                        )
                                        
                                        chunk_transcription = processor.batch_decode(
                                            predicted_ids, 
                                            skip_special_tokens=True
                                        )[0]
                                        
                                        if chunk_transcription.strip():
                                            transcriptions.append(chunk_transcription)
                                            
                                    except Exception as e:
                                        st.warning(f'Ошибка при обработке чанка {i+1}: {e}')
                                        continue
                            
                            transcription = " ".join(transcriptions)
                            progress_bar.empty()
                        
                        st.success(f'✅ Речь распознана!')
                        
                        # показываем распознанный текст
                        st.subheader('Распознанный текст:') 
                        st.text_area('Текст', transcription, height=300, key='transcription')
                        
                        # суммаризация текста с настройками
                        if len(transcription.strip()) > 100:
                            with st.spinner('Суммаризация текста...'):
                                @st.cache_resource
                                def load_summarizer():
                                    try:
                                        summarizer = pipeline(
                                            'summarization',
                                            model='IlyaGusev/mbart_ru_sum_gazeta',
                                            tokenizer='IlyaGusev/mbart_ru_sum_gazeta',
                                            device=0 if torch.cuda.is_available() else -1
                                        )
                                        return summarizer
                                    except Exception as e:
                                        st.warning(f'Не удалось загрузить суммаризатор: {e}')
                                        return None
                                
                                summarizer = load_summarizer()
                                
                                if summarizer is not None:
                                    # настройки суммаризации
                                    with st.expander('Настройки суммаризации ⚙️'):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            repetition_penalty = st.slider(
                                                'Штраф за повторения', 
                                                min_value=1.0, 
                                                max_value=5.0, 
                                                value=2.0,
                                                step=0.1
                                            )
                                        
                                        with col2:
                                            no_repeat_ngram_size = st.slider(
                                                'Запрет N-грамм', 
                                                min_value=1, 
                                                max_value=5, 
                                                value=3
                                            )
                                        
                                        with col3:
                                            num_beams = st.slider(
                                                'Лучевой поиск', 
                                                min_value=1, 
                                                max_value=8, 
                                                value=4
                                            )
                                        
                                        length_penalty = st.slider(
                                            'Штраф за длину', 
                                            min_value=0.5, 
                                            max_value=2.0, 
                                            value=1.0,
                                            step=0.1
                                        )

                                    try:
                                        # Суммаризация с гиперпараметрами
                                        summary = summarizer(
                                            transcription,
                                            max_length=800,
                                            min_length=80,
                                            repetition_penalty=repetition_penalty,
                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                            num_beams=num_beams,
                                            length_penalty=length_penalty,
                                            do_sample=False,
                                            truncation=True
                                        )[0]['summary_text']
                                        
                                        st.success('Текст суммаризирован!')
                                        st.subheader('Суммаризация:')
                                        st.info(summary)
                                        
                                        
                                    except Exception as e:
                                        st.warning(f'Ошибка суммаризации: {e}')

                        else:
                            st.warning('Текст слишком короткий для суммаризации')
                        
                        # Скачивание результатов
                        if 'summary' in locals():
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.download_button(
                                    '📥 Скачать файл с текстом',
                                    data=transcription,
                                    file_name='transcription.txt',
                                    mime='text/plain'
                                )
                            with col_d2:
                                st.download_button(
                                    '📥 Скачать файл с суммаризацией',
                                    data=summary,
                                    file_name='summary.txt',
                                    mime='text/plain'
                                )
                        else:
                            st.download_button(
                                '📥 Скачать текст',
                                data=transcription,
                                file_name='transcription.txt',
                                mime='text/plain'
                            )
                        
                    except Exception as e:
                        st.error(f'Ошибка: {str(e)}')
