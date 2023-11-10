# EchoPrompt 
[Link to Paper](https://arxiv.org/abs/2309.10687)

### Abstract
Language models are achieving impressive performance on various tasks by aggressively adopting inference-time prompting techniques, such as zero-shot and few-shot prompting. In this work, we introduce EchoPrompt, a simple yet effective approach that prompts the model to rephrase its queries before answering them. EchoPrompt is adapted for both zero-shot and few-shot in-context learning with standard and chain-of-thought prompting. Experimental results show that EchoPrompt yields substantial improvements across all these settings for four families of causal language models. These improvements are observed across various numerical reasoning (e.g. GSM8K, SVAMP), reading comprehension (e.g. DROP), and logical reasoning (e.g. Coin Flipping) tasks. On average, EchoPrompt improves the Zero-shot-CoT performance of code-davinci-002 by 5% in numerical tasks and 13% in reading comprehension tasks. We investigate the factors contributing to EchoPrompt's effectiveness through ablation studies, which reveal that both the original query and the model-generated rephrased version are instrumental in its performance gains. Our empirical results indicate that EchoPrompt is an effective technique that enhances in-context learning performance. We recommend incorporating EchoPrompt into various baseline prompting strategies to achieve performance boosts.


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
python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode zero_shot

## zero_shot+echoprompt  // (in implementation, zero_shot+echoprompt  is equivalent to zero_shot_cot+echoprompt with a different prompt)
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode zero_shot_cot --zero_shot_prompt "Let's repeat the question. \"" --max_tokens 300
python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode zero_shot_cot --max_tokens 300 --zero_shot_prompt "Let's repeat the complete question. \"" --zero_shot_prompt_stage2 "Therefore, in arabic numerals, the answer is"

## standard
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode standard --max_tokens 300
## standard+echoprompt
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode standard_rephrase_v1 --max_tokens 300
python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode standard --max_tokens 150
python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode standard_rephrase_v1 --max_tokens 300

## cot
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode cot --max_tokens 300
## cot+echoprompt
python main.py --model-name code-davinci-002 --dataset multiarith --learning_mode cot_rephrase_v1 --max_tokens 450

python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode cot --max_tokens 450
python main.py --model-name gpt-3.5-turbo-0301 --dataset drop_break --learning_mode cot_rephrase_v1 --max_tokens 600

<!-- on GSM8K dataset -->
## cot
python main.py --model-name gpt-3.5-turbo-0301 --dataset gsm8k --learning_mode cot --max_tokens 450
## cot+echoprompt
python main.py --model-name gpt-3.5-turbo-0301 --dataset gsm8k --learning_mode cot_rephrase_v1 --max_tokens 600

```

### Note
* For exact reproduction of the results, we encourage you to use either code-davinci-002, or gpt-3.5-turbo-instruct, (or one of the CodeLLama-instruct models) instead of gpt-3.5-turbo(since this model finetuned for chat applications, the model generates verbal responses and the answer extraction code is not suitable for this model). It would require some manual analysis, through the logs.
* As mentioned above, the results obtained with gpt-3.5-turbo on certain datasets (e.g GSM8K), may not align precisely with the values presented in the paper. This discrepancy arises because the model does not generate responses in the exact extraction format used in this code, since the model is finetuned for chat applications. Through human evaluation, we observed that both with and without EchoPrompt, the model achieved higher accuracies. However, it's worth noting that even though there may be such minor variations of around 1-2% compared to the paper's results, the improvements of Echoprompt should be clearly differentiable. Additionally, we did not encounter similar issues with other models.
* All the experimental values reported in Table-9 in the paper are `averaged` over 3 iterations.
* Logs of the experiments can be found in the `logs` folder
* Final results of the experiments are saved in the `saved_results` folder
* The `max_tokens` parameter should be adjusted based on the task's complexity. More complex tasks, which necessitate longer explanations, will require a higher number of tokens. 

