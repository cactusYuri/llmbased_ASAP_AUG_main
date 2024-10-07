"""
python do_aug.py --input ./train.tsv --output_dir ./outdata --mode word --method tfidf --testlevel 0
python do_aug.py --input ./train.tsv --output_dir ./outdata --mode word --testlevel 1
python do_aug.py --input ./train.tsv --output_dir ./outdata --testlevel 2
"""

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from nlpaug.util import Action
import os
import argparse
import re
import pandas as pd
from typing import List, Dict, Callable
import nltk
import time
import csv

ADVANCED_TOKENIZER_REGEX = re.compile(r'(\b\w+[\w\']*\w+\b|\b\w+\b|[^\w\s]|\s+)')
def advanced_tokenize(text):
    return [token for token in re.findall(ADVANCED_TOKENIZER_REGEX, text) if token.strip()]
protected_pattern = r'@\w+'
device = 'cpu'
# 定义增强方法字典
AUGMENTERS: Dict[str, Dict[str, Callable]] = {
    'word': {
        'synonym': lambda: naw.SynonymAug(aug_src='wordnet',stopwords_regex=protected_pattern),
        'antonym': lambda: naw.AntonymAug(aug_max=2,stopwords_regex=protected_pattern),
        'random': lambda: naw.RandomWordAug(action='substitute', aug_min=1, aug_max=2,stopwords_regex=protected_pattern),
        'split': lambda: naw.SplitAug(aug_min=1, aug_max=2,stopwords_regex=protected_pattern),
        'backTranslation': lambda: naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en',
            device=device
        ),
        'bert': lambda: naw.ContextualWordEmbsAug(
            model_path='bert-base-cased', action="substitute",aug_max=2, device=device,stopwords_regex=protected_pattern
        ),
        'glove': lambda: naw.WordEmbsAug(
            model_type='glove', model_path='./glove.6B.50d.txt',aug_max=2,stopwords_regex=protected_pattern
        ),
        'fasttext': lambda: naw.WordEmbsAug(
            model_type='fasttext', model_path='./wiki-news-300d-1m.vec',aug_max=2,stopwords_regex=protected_pattern
        ),
        'word2vec': lambda: naw.WordEmbsAug(
            model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin',aug_max=2,stopwords_regex=protected_pattern
        ),
        'tfidf': lambda: naw.TfIdfAug(
            model_path='./data',aug_min=1,aug_max=2,
            aug_p=0.3,stopwords_regex=protected_pattern
        ),
    },
    'char': {
        'ocr': lambda: nac.OcrAug(stopwords_regex=protected_pattern, tokenizer=advanced_tokenize,aug_char_max=1,aug_word_max=3),
        'keyboard': lambda: nac.KeyboardAug(stopwords_regex=protected_pattern, tokenizer=advanced_tokenize,aug_char_max=1,aug_word_max=3),
        'random': lambda: nac.RandomCharAug(stopwords_regex=protected_pattern, tokenizer=advanced_tokenize,aug_char_max=1,aug_word_max=3),
    },
    'sentence': {
        'random': lambda: nas.RandomSentAug(),
        'gpt2': lambda: nas.ContextualWordEmbsForSentenceAug(
            model_path='gpt2'
        ),
    }
}

def load_data(file_path: str) -> pd.DataFrame:
    if os.path.splitext(file_path)[1]=='.xlsx':
        return pd.read_excel(file_path, header=None)
    else:
        return pd.read_csv(file_path, header=None, sep='\t')
    
def save_data(data: pd.DataFrame, file_path: str):
    if os.path.splitext(file_path)[1]=='.xlsx':
        data.to_excel(file_path, index=False, header=None)
    else:
        data.to_csv(file_path, index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)

def augment_full_data(augmenter, data: pd.Series) -> pd.Series:
    return data.apply(lambda x: augmenter.augment(x)[0])

def TBaugment_data(augmenter, data: pd.Series) -> pd.Series:
    def augment_article(article: str) -> str:
        start_time = time.time()    
        # 分解文章为句子
        sentences: List[str] = nltk.sent_tokenize(article)
        
        augmented_parts: List[str] = []
        current_part: List[str] = []
        current_length: int = 0
        
        for sentence in sentences:
            current_part.append(sentence)
            current_length += len(sentence)
            
            if current_length > 300:
                # 对累积的句子进行增强
                part_to_augment = ' '.join(current_part)
                augmented_part = augmenter.augment(part_to_augment)[0]
                augmented_parts.append(augmented_part)
                
                # 重置当前部分
                current_part = []
                current_length = 0
        
        # 处理剩余的句子
        if current_part:
            part_to_augment = ' '.join(current_part)
            augmented_part = augmenter.augment(part_to_augment)[0]
            augmented_parts.append(augmented_part)
        
        # 重新组合成文章
        new_article = ' '.join(augmented_parts)
        end_time = time.time()
        print(f"\r{augmenter.name} Augmented data Time taken: {(end_time - start_time):5.2f} seconds", end='', flush=True)
        return new_article
    
    return data.apply(augment_article)

def augment_data(augmenter, data: pd.Series) -> pd.Series:
    def augment_article(article: str) -> str:
        start_time = time.time()    
        # 分解文章为句子
        sentences: List[str] = nltk.sent_tokenize(article)
        
        # 对每个句子进行增强
        augmented_sentences: List[str] = [augmenter.augment(sent)[0] for sent in sentences]
        
        # 重新组合成文章
        new_article = ' '.join(augmented_sentences)
        end_time = time.time()
        print(f"\r{augmenter.name} Augmented data Time taken: {(end_time - start_time):5.2f} seconds", end='', flush=True)
        return new_article
    
    return data.apply(augment_article)

def generate_output_filename(input_file: str, aug_mode: str, aug_method: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if os.path.splitext(input_file)[1]=='.xlsx':
        return f"{base_name}_{aug_mode}_{aug_method}_augmented.xlsx"
    else:
        return f"{base_name}_{aug_mode}_{aug_method}_augmented"+os.path.splitext(input_file)[1]

def process_augmentation(input_file: str, aug_mode: str, aug_method: str, output_dir: str):
    data = load_data(input_file)
    start_time = time.time()    
    augmenter = AUGMENTERS[aug_mode][aug_method]()
    if aug_method=='back_translation':
        data.iloc[:, 2] = TBaugment_data(augmenter, data.iloc[:, 2])
    else:
        data.iloc[:, 2] = augment_data(augmenter, data.iloc[:, 2])
    end_time = time.time()
    output_file = os.path.join(output_dir, generate_output_filename(input_file, aug_mode, aug_method))
    print(f"\n{output_file} Time taken: {(end_time - start_time):.2f} seconds")
    save_data(data, output_file)
    print(f"Augmented data saved to {output_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Text data augmentation")
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--mode", choices=['word', 'char', 'sentence'], help="Augmentation mode")
    parser.add_argument("--method", help="Augmentation method")
    parser.add_argument("--testlevel", type=int, choices=[0, 1, 2], default=0, help="Test level")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.testlevel == 0:
        if not args.mode or not args.method:
            raise ValueError("Mode and method must be specified for testlevel 0")
        process_augmentation(args.input, args.mode, args.method, args.output_dir)
    
    elif args.testlevel == 1:
        if not args.mode:
            raise ValueError("Mode must be specified for testlevel 1")
        for aug_method in AUGMENTERS[args.mode]:
            process_augmentation(args.input, args.mode, aug_method, args.output_dir)
    
    elif args.testlevel == 2:
        for aug_mode in AUGMENTERS:
            for aug_method in AUGMENTERS[aug_mode]:
                #if aug_method=='backTranslation':
                #    continue
                process_augmentation(args.input, aug_mode, aug_method, args.output_dir)

if __name__ == "__main__":
    main()