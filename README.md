# EchoPrompt

### Install dependencies
```
<!-- Tested with python 3.9 -->
pip install -r requirements.txt
```

### Credentials
* create openai_key.txt file in root folder and add your openai api-key (sk-***)

### Run an experiment
```
<!-- on MultiArith dataset -->
## zero_shot
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode zero_shot
## zero_shot+echoprompt  // (in implementation, zero_shot+echoprompt  is equivalent to zero_shot_cot+echoprompt with a different prompt)
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode zero_shot_cot --zero_shot_prompt "Let's repeat the question. \"" --max_tokens 300

## standard
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode standard
## standard+echoprompt
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode standard_rephrase_v1

## cot
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode cot
## cot+echoprompt
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode cot_rephrase_v1

```

### Note
* Logs of the experiments can be found in the logs folder
* Final results of the experiments are saved in the saved_results folder