from argparse import ArgumentParser
import codecs
import csv
import gc
import random
import os
import shutil

from datasets import load_dataset
from datasets.features import Audio
from jiwer import wer
import numpy as np
from scipy.io.wavfile import write
from transformers import T5ForConditionalGeneration, Qwen2ForCausalLM
from transformers import GenerationConfig
from transformers import T5Tokenizer, Qwen2Tokenizer
import torch
from tqdm import tqdm, trange


RANDOM_SEED: int = 42
TARGET_SAMPLING_RATE: int = 16_000
MIN_TEXT_SIZE: int = 3


def restore_text_with_t5(text: str, tokenizer: T5Tokenizer, config: GenerationConfig,
                         model: T5ForConditionalGeneration) -> str:
    if len(text) == 0:  # if an input text is empty, then we return an empty text too
        return ''
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 2.0 + 10)
    min_size = MIN_TEXT_SIZE
    if x.input_ids.shape[1] <= min_size:
        return text
    out = model.generate(**x, generation_config=config, max_length=max_size)
    res = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return ' '.join(res.split())


def restore_text_with_meno(text: str, tokenizer: Qwen2Tokenizer, config: GenerationConfig,
                           model: Qwen2ForCausalLM) -> str:
    if len(text) == 0:  # if an input text is empty, then we return an empty text too
        return ''
    messages = [
        {"role": "system", "content": "Исправь, пожалуйста, ошибки распознавания речи в следующем тексте, восстанови в нём знаки пунктуации и правильно расставь прописные и строчные буквы. Пиши свой ответ грамотно, с учётом морфологии и синтаксиса русского языка."},
        {"role": "user", "content": "вы выборском районе города проводится проверка по факту нападению на квартиру"},
        {"role": "assistant", "content": "В Выборгском районе города проводится проверка по факту нападения на квартиру."},
        {"role": "user", "content": "в пораде победы на красной площади в москве премут участие свысше четырнадцать ты своим наслужащих в том числе одинствого музыканта всводного оркестра сообщил интерфакт словнвы четверг представителе правления прес службы и информацию именно борон рф по сухоопутным войскам полковник сергей власов"},
        {"role": "assistant", "content": "В параде Победы на Красной площади в Москве примут участие свыше 14 тыс. военнослужащих, в том числе 1 100 музыкантов сводного оркестра, сообщил «Интерфаксу-АВН» в четверг представитель управления пресс-службы и информации Минобороны РФ по Сухопутным войскам полковник Сергей Власов."},
        {"role": "user", "content": "сбер где поблизости можно воспользоваться банкоматом чтобы оплатить услуги"},
        {"role": "assistant", "content": "Сбер, где поблизости можно воспользоваться банкоматом, чтобы оплатить услуги?"},
        {"role": "user", "content": "шестнадцатая часть сезона пять сериала лемони сникет тридцать три несчастья"},
        {"role": "assistant", "content": "16-я часть сезона 5 сериала «Лемони Сникет: 33 несчастья»."},
        {"role": "user", "content": "главные потребители дискофс виндовс ими крософтов фикироссийские сборщики устанавливающие программные обеспечение прана выпускаемые ими компьютеры и серверы отсюда название а инверсия этой яверсия для оригиналгикоепментмано фактуер для сборщика"},
        {"role": "assistant", "content": "Главные потребители дисков с «Windows» и «Microsoft Office» - российские сборщики, устанавливающие программное обеспечение (ПО) на выпускаемые ими компьютеры и серверы (отсюда название OEM-версия, т. е. версия для «Original Equipment Manufacturer», для сборщика)."},
        {"role": "user", "content": "в две тысячи тринадцать год уконкурс гуглеский енки фаир организатором которого выступает компания гугле проводится в этретий раз веконку всемогут участвовать деть в возрасти от тринадцать да восемнадцать лет свои научные проекты участники от правляют на рассмотрения через интернетых изучает жури состоящие из ученых и сотрудников гогль он уже определяет девяносто региональных изатем пятнадцать глобальных феналистов"},
        {"role": "assistant", "content": "В 2013 году конкурс Google Science Fair, организатором которого выступает компания Google, проводится в третий раз. В конкурсе могут участвовать дети в возрасте от 13 до 18 лет. Свои научные проекты участники отправляют на рассмотрение через интернет. Их изучает жюри, состоящее из ученых и сотрудников Google. Оно же определяет 90 региональных, а затем 15 глобальных финалистов."},
        {"role": "user", "content": text}
    ]
    text_with_fewshots = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    max_size = len(tokenizer.tokenize(text)) * 2 + 10
    x = tokenizer([text_with_fewshots], return_tensors="pt").to(model.device)
    out = model.generate(**x, generation_config=config, max_new_tokens=max_size)[0]
    res = tokenizer.decode(out[len(x.input_ids[0]):], skip_special_tokens=True)
    return ' '.join(res.split()).strip()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available!')
    torch.cuda.random.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input ASR dataset name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output ASR dataset name.')
    parser.add_argument('--t5', dest='t5_name', type=str, required=False,
                        default='bond005/ruT5-ASR-large', help='The path to the T5-based ASR corrector.')
    parser.add_argument('--meno', dest='meno_name', type=str, required=False,
                        default='bond005/meno-tiny-0.1', help='The Meno-based ASR corrector name.')
    args = parser.parse_args()

    target_ds_name = os.path.normpath(args.output_name)
    if not os.path.isdir(target_ds_name):
        os.mkdir(target_ds_name)
    if not os.path.isdir(target_ds_name):
        raise IOError(f'The directory "{target_ds_name}" does not exist!')

    t5_tokenizer_for_restoring = T5Tokenizer.from_pretrained(args.t5_name)
    t5_model_for_restoring = T5ForConditionalGeneration.from_pretrained(
        args.t5_name,
        torch_dtype=torch.float32,
        device_map='cuda:0',
        attn_implementation='eager'
    )
    t5_config_for_restoring = GenerationConfig.from_pretrained(args.t5_name)
    t5_model_for_restoring.eval()
    print(f'The T5-based ASR corrector is loaded.')

    meno_model_for_restoring = Qwen2ForCausalLM.from_pretrained(
        args.meno_name,
        torch_dtype=torch.float16,
        device_map='cuda:0',
        attn_implementation='sdpa'
    )
    meno_model_for_restoring.eval()
    meno_config_for_restoring = GenerationConfig.from_pretrained(args.meno_name)
    meno_tokenizer_for_restoring = Qwen2Tokenizer.from_pretrained(args.meno_name)
    print(f'The Meno-based ASR corrector is loaded.')

    source_dataset = load_dataset(args.input_name, keep_in_memory=True, streaming=False,
                                  num_proc=max(1, os.cpu_count()))
    ds_parts = ['train', 'validation', 'test']
    if os.path.isdir(os.path.join(target_ds_name, 'data')):
        shutil.rmtree(os.path.join(target_ds_name, 'data'))
    if not os.path.isdir(os.path.join(target_ds_name, 'data')):
        os.mkdir(os.path.join(target_ds_name, 'data'))
    if not os.path.isdir(os.path.join(target_ds_name, 'data')):
        raise IOError(f'The directory "{os.path.join(target_ds_name, "data")}" does not exist!')
    counter = 1
    with codecs.open(os.path.join(target_ds_name, 'metadata.csv'), mode='w', encoding='utf-8', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['file_name', 'transcription', 'inv_normalized_1', 'inv_normalized_2',
                              'wer_based_uncertainty'])
        for cur_part in ds_parts:
            if cur_part in source_dataset:
                print(f'\nThe {cur_part} part of the {os.path.basename(args.input_name)} dataset is processed.')
                if source_dataset[cur_part].features['audio'].sampling_rate == TARGET_SAMPLING_RATE:
                    current_data = source_dataset[cur_part]
                else:
                    current_data = source_dataset[cur_part].cast_column(
                        'audio',
                        Audio(sampling_rate=TARGET_SAMPLING_RATE)
                    )
                    print('Audio data are resampled.')
                print(f'There are {len(current_data)} samples before filtering.')
                sounds = []
                transcriptions = []
                for idx in trange(len(current_data)):
                    current_sample = current_data[idx]
                    if (current_sample['audio'] is not None) and (current_sample['transcription'] is not None):
                        new_sound = current_sample['audio']['array']
                        if new_sound is not None:
                            if isinstance(new_sound, np.ndarray):
                                if (len(new_sound.shape) == 1) and (new_sound.shape[0] > TARGET_SAMPLING_RATE):
                                    if (np.max(new_sound) > 1.0) or (np.min(new_sound) < -1.0):
                                        err_msg = f'The sample {idx} of the {cur_part} part contains a wrong sound!'
                                        raise IOError(err_msg)
                                    new_transcription = ' '.join(
                                        current_sample['transcription'].strip().split()
                                    ).strip()
                                    if len(new_transcription) > MIN_TEXT_SIZE:
                                        sounds.append(np.asarray(np.round(new_sound * 32767.0), dtype=np.int16))
                                        transcriptions.append(new_transcription)
                    del current_sample
                if len(sounds) < 1:
                    err_msg = (f'There are no good samples in the {cur_part} part of '
                               f'the {os.path.basename(args.input_name)} dataset.')
                    raise IOError(err_msg)
                print(f'There are {len(sounds)} samples after filtering.')
                del current_data

                samples = []
                for idx, val in enumerate(tqdm(transcriptions)):
                    predicted_with_t5 = restore_text_with_t5(val, t5_tokenizer_for_restoring,
                                                             t5_config_for_restoring, t5_model_for_restoring)
                    predicted_with_meno = restore_text_with_meno(val, meno_tokenizer_for_restoring,
                                                                 meno_config_for_restoring, meno_model_for_restoring)
                    uncertainty = max(
                        wer(predicted_with_t5, predicted_with_meno),
                        wer(predicted_with_meno, predicted_with_t5)
                    )
                    samples.append((
                        sounds[idx],         # 0
                        val,                 # 1
                        predicted_with_t5,   # 2
                        predicted_with_meno, # 3
                        uncertainty          # 4
                    ))
                    del predicted_with_t5, predicted_with_meno
                del sounds, transcriptions
                gc.collect()
                torch.cuda.empty_cache()

                if os.path.isdir(os.path.join(target_ds_name, 'data', cur_part)):
                    shutil.rmtree(os.path.join(target_ds_name, 'data', cur_part))
                if not os.path.isdir(os.path.join(target_ds_name, 'data', cur_part)):
                    os.mkdir(os.path.join(target_ds_name, 'data', cur_part))
                if not os.path.isdir(os.path.join(target_ds_name, 'data', cur_part)):
                    raise IOError(f'The directory "{os.path.join(target_ds_name, "data", cur_part)}" does not exist!')
                uncertainties = []
                for cur_sample in samples:
                    new_sound_name = 'data/{0}/sound{1:>06}.wav'.format(cur_part, counter)
                    new_sound_name_ = os.path.join(target_ds_name, 'data', cur_part, 'sound{0:>06}.wav'.format(counter))
                    counter += 1
                    data_writer.writerow(
                        [new_sound_name, cur_sample[1], cur_sample[2], cur_sample[3], round(cur_sample[4], 4)]
                    )
                    write(new_sound_name_, rate=TARGET_SAMPLING_RATE, data=cur_sample[0])
                    uncertainties.append(cur_sample[4])
                uncertainties.sort()
                info_msg = ('Uncertainties of the inverse normalization: minimal = {0:.4f}, maximal = {1:.4f}, '
                            'mean = {2:.4f}, median = {3:.4f}.').format(
                    min(uncertainties), max(uncertainties), np.mean(uncertainties),
                    uncertainties[(len(uncertainties) - 1) // 2]
                )
                del samples
                gc.collect()
                print(info_msg)
                del uncertainties


if __name__ == '__main__':
    main()
