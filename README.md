# mdok
mdok detector for the "Voight-Kampff" Generative AI Authorship Verification at PAN 2025

The code for mdok-binary inference is available in [repo](https://github.com/DominikMacko/mdok).
The model adapter of mdok-binary is available at [mdok HuggingFace](https://huggingface.co/DominikMacko/mdok).

## mdok Training

For mdok-multiclas, run the provided [mdok-multiclass.py](https://github.com/kinit-sk/mdok/blob/main/mdok-multiclass.py) with the following arguments:
```
python mdok-multiclass.py --train_file_path "clef/train.jsonl" --dev_file_path "clef/dev.jsonl" --test_file_path "clef/test.jsonl" --model "Qwen/Qwen3-4B-Base" --prediction_file_path "test_predictions.csv"
```

## Cite
If you use the model, code, data, or any information from this repository, please cite the paper(s):
```
@misc{macko2025mdokkinitrobustlyfinetuned,
      title={mdok of {KInIT}: Robustly Fine-tuned {LLM} for Binary and Multiclass {AI}-Generated Text Detection}, 
      author={Dominik Macko},
      year={2025},
      eprint={2506.01702},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01702}, 
}
@misc{macko2025increasingrobustnessfinetunedmultilingual,
      title={Increasing the Robustness of the Fine-tuned Multilingual Machine-Generated Text Detectors}, 
      author={Dominik Macko and Robert Moro and Ivan Srba},
      year={2025},
      eprint={2503.15128},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15128}, 
}
```
