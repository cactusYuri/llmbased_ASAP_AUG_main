# Experimental Code for Paper - Text Augmentation and Automated Scoring

This project contains code for text augmentation and automated scoring used in the paper.

## Project Structure

- `augByNlpaug/do_aug.py`: Text augmentation using the nlpaug library
- `augByNlpaug/do_aug2.py`: Another version of text augmentation using the nlpaug library
- `eda_nlp-master/eda_nlp-master/code/eda.py`: Text augmentation implementation based on EDA
- `generate.py`: Core processing program for data augmentation based on optimized AI
- `trains.py`: Training and evaluation program modified for Essay Scoring
- `test_final_4/`: Final data summary analysis

## Instructions

1. Run `augByNlpaug/do_aug.py` to generate nlpaug augmented data
   Requires the `nlpaug` library
   - `testlevel 0`: Generate augmented data for specified mode and method
   - `testlevel 1`: Generate all augmented data for specified mode
   - `testlevel 2`: Generate all augmented data for all modes
   
   ```
   python do_aug.py --input ./train.tsv --output_dir ./outdata --mode word --method tfidf --testlevel 0
   python do_aug.py --input ./train.tsv --output_dir ./outdata --mode word --testlevel 1
   python do_aug.py --input ./train.tsv --output_dir ./outdata --testlevel 2
   ```

2. Run `eda_nlp-master/eda_nlp-master/code/eda.py` to generate EDA augmented data

3. Run `generate.py` to generate optimized AI augmented data

4. Run `trains.py` for model training and evaluation
   - Download the open-source Essay Scoring model and copy `trains.py` to the root directory of Essay Scoring
   - Place the data generated by steps 1, 2, and 3 in the `datas/` directory, naming them as follows: `train_eda_augmented.tsv`, `train_ai_augmented.tsv`, `train_{method}_augmented.tsv`
   
   ```
   python trains.py --embedding glove --embedding_dict glove.6B.50d.txt --datapath ./datas/ --oov embedding --prompt_id 1
   ```
   - Scan the data files in the specified `datapath` directory and use datasets named `train_{method}_augmented.tsv` for training. Then use `dev.tsv` for evaluation and `test.dev` for validation to obtain the optimal values of "QWK", "Pearson", "Spearman" for each `{method}` and the average values across multiple epochs.

'''  result
+------------------+------------+------------+------------+------------+------------+------------+
| Method           | Best       |            |            | Avg        |            |            |
|                  | QWK        | Pearson    | Spearman   | QWK        | Pearson    | Spearman   |
+------------------+------------+------------+------------+------------+------------+------------+
| 0org             | 0.74538334 | 0.77946597 | 0.99315760 | 0.66436193 | 0.75320494 | 0.99672133 |
| 1llm-based       | 0.81954922 | 0.82781173 | 0.99892291 | 0.75128993 | 0.78552401 | 0.99636469 |
| 2char_keyboard   | 0.79700103 | 0.80889189 | 0.99225794 | 0.73914048 | 0.79775757 | 0.99580305 |
| 3char_ocr        | 0.76538303 | 0.78297317 | 0.99476695 | 0.71320085 | 0.77692246 | 0.99607605 |
| 4word_antonym    | 0.79244062 | 0.81333327 | 0.99320286 | 0.72941322 | 0.79280335 | 0.99589781 |
| 5word_bert       | 0.76190115 | 0.78477484 | 0.99653340 | 0.71473679 | 0.77280408 | 0.99596886 |
| 6word_synonym    | 0.79241085 | 0.81405485 | 0.99459908 | 0.72906566 | 0.79998040 | 0.99581856 |
| 7word_glove      | 0.80822706 | 0.81802940 | 0.99882120 | 0.74600303 | 0.80092651 | 0.99605152 |
| 8sentence_random | 0.78097594 | 0.80023968 | 0.99613551 | 0.72444660 | 0.79657549 | 0.99663995 |
| 9sentence_gpt2   | 0.73632073 | 0.77010959 | 0.99812446 | 0.66975185 | 0.75987566 | 0.99535937 |
| 10eda            | 0.75645322 | 0.78129142 | 0.99728561 | 0.73746911 | 0.75241791 | 0.99816346 |
+------------------+------------+------------+------------+------------+------------+------------+
```
5. Use the scripts in `test_final_4/` for final data analysis

## Dependencies

- Python 3.6+
- nlpaug
- PyTorch
- Other dependencies are listed in `requirements.txt`

## Citation

If you use the code from this project, please cite the following paper:

[Your paper citation information]

## Acknowledgements

- [EDA](https://github.com/jasonwei20/eda_nlp)
- [Essay-Scoring](https://github.com/zifengcheng/Essay-Scoring)

## License

[Choose an appropriate open source license]