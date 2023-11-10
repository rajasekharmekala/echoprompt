import os
import copy
import time
import logging
import random

import argparse
from tqdm import tqdm
import math
from statistics import mean

from utils.dataset import get_questions, remove_char, is_equivalent, dates_equal
from utils.api import  get_responses, is_single_prompt_model
from utils.responses import get_final_response, get_final_response_zero_shot, get_final_response_options_later


logger = logging.getLogger(__name__)

class Opts(object):
    def __init__(self, my_dict):      
        for key in my_dict:
            setattr(self, key, my_dict[key])

def add_prompt(question_list, prompt_instruction, question_prepend, answer_prepend):
    templated_question = copy.deepcopy(question_list)
    for question in templated_question:
        question['golden_question'] = question['question']
        question['question'] = prompt_instruction + "\n" + \
            question_prepend + question['question'] + "\n" + answer_prepend
        question['question'] = question['question'].strip("\n")
    return templated_question

def add_instruction_prompt(question_list, prompt_instruction, question_prepend, answer_prepend, learning_mode):
    templated_question = copy.deepcopy(question_list)
    for question in templated_question:
        question['golden_question'] = question['question']
        # question['question'] = f"""<s>[INST] <<SYS>>\n{args.zero_shot_prompt} Finally, conclude with the phrase: "{args.zero_shot_prompt_stage2}", followed by the answer.\n<</SYS>>\n{question['question']} [/INST] """
        question['question'] = f"""<s>[INST] <<SYS>><</SYS>>\n{question['question']}\n{args.zero_shot_prompt} [/INST] """
    return templated_question

def make_batch(question_list, batch_size):
    batches = []
    for i in range(0, len(question_list), batch_size):
        batches.append(question_list[i:i+batch_size])
    return batches



def run_queries(args, api_keys, api_base, templated_questions, number_of_samples, max_tokens, prev_total, prev_correct, retry=0, initial_batch_size=20):
    DELAY = 10
    if 'least_to_most' in args.learning_mode:
        DELAY = 15

    if(is_single_prompt_model(args)):
        DELAY = 1

    correct = prev_correct
    total = prev_total
    failures = []
    completed = []
    if not is_single_prompt_model(args):
        batch_size = int(initial_batch_size/(1+retry))
    else:
        batch_size = initial_batch_size

    batches = make_batch(templated_questions[:number_of_samples], batch_size)
    loop = tqdm(batches, total=math.ceil(len(templated_questions[:number_of_samples])/batch_size))

    if args.learning_mode == 'least_to_most':
        path = f'prompts/{args.dataset}/least_to_most_prompt.txt' if os.path.isfile(f'prompts/{args.dataset}/least_to_most_prompt.txt') else f'prompts/least_to_most_prompt.txt'
        with open(path, 'r') as f:
            stage2_prompt = f.read()

    for b_idx, data_batch in enumerate(loop):

        q_per_process = int(len(data_batch)/len(api_keys))
        if args.learning_mode == 'zero_shot_cot' or args.learning_mode == 'zero_shot' or args.learning_mode == 'zero_shot_augment' or args.learning_mode == 'zero_shot_cot_augment':
            responses = get_final_response_zero_shot(args.learning_mode == 'zero_shot_cot' or args.learning_mode == 'zero_shot_cot_augment', "\"" in args.zero_shot_prompt,  data_batch, f" {args.zero_shot_prompt_stage2}", api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=DELAY)
        elif args.learning_mode == 'least_to_most':
            responses = get_final_response(data_batch, stage2_prompt, api_keys, api_base, model_name, max_tokens, q_per_process, args)
        elif 'cot_options_later' in args.learning_mode:
            responses = get_final_response_options_later(data_batch, f" {args.zero_shot_prompt_stage2}", api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=DELAY)
            
        else:
            input_query = [question['question'] for question in data_batch]
            responses = get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args)

        if(len(responses)==0):
            continue

        for i, answer in enumerate(responses):
            full_answer = answer['text']
            id = answer["index"]
            stop_reason = answer["finish_reason"]
            is_correct = False
            final_answer = ''
            question_sample = data_batch[id]
            if args.learning_mode == 'cot_simplify':
                last_answer = full_answer.split("rewriting in simple words, the question is: ")
            elif args.dataset == 'coin_flip' and 'zero_shot' in args.learning_mode:
                last_answer = full_answer.lower().split("the answer (yes or no) is \"")
            else:
                last_answer = full_answer.lower().split("the answer is")
        
            if stop_reason == 'stop'or len(last_answer) >= 2:
                if args.learning_mode == 'cot_simplify':
                    is_correct = True
                    correct = correct + 1
                    final_answer = last_answer[-1].strip().strip('"')
                    question_sample['complete'] = True
                    total = total + 1
                    completed.append(question_sample)
                else:

                    logger.info("question:\n" + data_batch[id]['question'])
                    logger.info("\nsubq: " + data_batch[id].get('sub_q',''))
                    logger.info("golden_answer:" +
                                str(data_batch[id]['golden_answer']))
                    final_answer = last_answer[-1].strip()
                    if len(final_answer) >0 and final_answer[-1] == '.':
                        final_answer = final_answer[:-1]
                    try:
                        if args.dataset == 'aqua' or args.dataset == 'mathqa' or args.dataset == 'mmlu_ele' or args.dataset == 'mmlu_high' or args.dataset == 'logiqa' or args.dataset == 'race_m' or args.dataset == 'race_h':
                            s = final_answer
                            if s[s.find("(")+1:s.find(")")] == data_batch[id]['golden_answer']:
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")
                        elif args.dataset == 'squad' or args.dataset == 'squadv2' or args.dataset == 'newsqa':
                            s = final_answer
                            pred_value = s.strip('"').strip().strip(".")
                            # print(s, "->", pred_value, "...")
                            if pred_value in data_batch[id]['golden_answer']:
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")     
                        elif args.dataset == 'shuffled_objects':
                            s = final_answer
                            pred_value = s.strip('"').strip().strip(".")
                            # print(s, "->", pred_value, "...")

                            if len(pred_value)>=3 and len(pred_value)<=20 and pred_value in data_batch[id]['golden_answer'] or  data_batch[id]['golden_answer'] in pred_value:
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")
                        elif args.dataset == 'scan':
                            final_answer = final_answer.replace('"', ' "')
                            final_answer = eval(final_answer).strip(" ")
                            # print('-'*20)
                            # print(final_answer)
                            # print(data_batch[id]['golden_answer'])
                            # print(final_answer==data_batch[id]['golden_answer'])
                            # print('-'*20)
                            if final_answer == data_batch[id]['golden_answer']:
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")

                        elif args.dataset == 'last_letter' or args.dataset == 'first_letter' or args.dataset == 'strategyqa'  or args.dataset == 'winogrande_xl' or args.dataset == 'coin_flip' or args.dataset == 'coin_flip_4' or args.dataset == 'csqa' or args.dataset == 'smcalflow':
                            final_answer = final_answer.strip("\"")
                            final_answer=final_answer.split('.')[0]
                            if args.dataset == 'coin_flip':
                                final_answer=final_answer.split(',')[0]
                            # print(last_answer, '---', final_answer,'---', data_batch[id])
                            if final_answer == data_batch[id]['golden_answer']:
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")
                        elif args.dataset == 'date_understanding':
                            final_answer = final_answer.strip("\"")
                            final_answer=final_answer.split('.')[0]
                            # print(last_answer, '---', final_answer,'---', data_batch[id])
                            if dates_equal(final_answer, data_batch[id]['golden_answer']):
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")

                        elif '_no_numbers' in args.dataset:
                            if  is_equivalent(final_answer.upper(), data_batch[id]['golden_answer']):
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")
                        elif args.dataset == 'count_objects':
                            if(final_answer.isdigit()):
                                if float(remove_char(final_answer)) == float(remove_char(data_batch[id]['golden_answer'][1])):
                                    logger.info("prediction: TRUE")
                                    correct = correct + 1
                                    is_correct = True
                                else:
                                    logger.info("prediction: FALSE")
                            else:
                                s = final_answer
                                pred_value = s.strip('"').strip().strip(".")
                                # print(s, "->", pred_value, "...")
                                if pred_value == data_batch[id]['golden_answer'][0]:
                                    logger.info("prediction: TRUE")
                                    correct = correct + 1
                                    is_correct = True
                                else:
                                    logger.info("prediction: FALSE")    

                        else:
                            if float(remove_char(final_answer)) == float(remove_char(data_batch[id]['golden_answer'])):
                                logger.info("prediction: TRUE")
                                correct = correct + 1
                                is_correct = True
                            else:
                                logger.info("prediction: FALSE")
                    except Exception as err:
                        print("*"*75)
                        print(err)
                        print(final_answer)
                    logger.info("The final answer: " + str(final_answer))
                    # logger.info("question:\n" + data_batch[id]['question'])
                    logger.info("The full answer:" + full_answer)
                    total = total + 1
                    question_sample['complete'] = True
                    logger.info("*************************")

                    completed.append(question_sample)

            else :
                logger.info("********FAILED*****************")
                logger.warning("question: " + data_batch[id]['question'])
                logger.warning("failed answer: " + answer['text'])
                question_sample['complete'] = False
                failures.append(question_sample)
                # print("stop_reason", stop_reason)
            
            question_sample['full_answer'] = full_answer
            question_sample['is_correct'] = is_correct
            question_sample['stop_reason'] = stop_reason
            question_sample['final_answer'] = final_answer
            if question_sample['complete']:
                question_logger.info(question_sample)

        # print(response)
        loop.set_description(f"correct: {correct}, total:{total}, failures:{len(failures)}")
        loop.set_postfix({'accuracy': correct/(total + len(failures))})

        if b_idx!= len(batches)-1:
            # print(DELAY)
            time.sleep(DELAY)
    
    return completed, failures, correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default="code-davinci-002", choices=[ 
                                                             "code-davinci-002",  "text-davinci-002", 
                                                             "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-instruct-0914", "gpt-3.5-turbo-instruct",
                                                             "gpt-4", 
                                                             "codellama/CodeLlama-7b-Instruct-hf", "codellama/CodeLlama-13b-Instruct-hf", "codellama/CodeLlama-34b-Instruct-hf",
                                                             ])
    
    #add more datasets as needed
    parser.add_argument('--dataset', type=str,
                        default="gsm8k", choices=["gsm8k", "svamp", "multiarith", "singleop",  "aqua", 
                                                  "date_understanding", "coin_flip",
                                                  "mathqa", "mmlu_ele", "mmlu_high",  
                                                  "squad", "drop_break", "drop_census",
                                                  "strategyqa", "csqa", "winogrande_xl",
                                                  "shuffled_objects",
                                                  ])
    parser.add_argument('--zero_shot_prompt', default="Let's think step by step.")
    parser.add_argument('--zero_shot_prompt_stage2', default="Therefore, the answer is")
    parser.add_argument('--max_tokens', type=int)
    parser.add_argument('--options_later', action='store_true') # no options are provided during text generation. 

    parser.add_argument('--learning_mode', type=str, default='cot', choices=[
        'standard', 'cot', 'zero_shot', 'zero_shot_cot',
        'cot_qrepeat', 'cot_rephrase', 'cot_rephrase_v1', 'cot_rephrase_v2', 'cot_rephrase_v3',
        'cot_options_later', 'cot_options_later_rephrase_v1',
        'standard_qrepeat', 'standard_rephrase', 'standard_rephrase_v1', 'standard_rephrase_v2', 'standard_rephrase_v3',
                        'least_to_most', 'least_to_most_2step', 'least_to_most_1step', 'least_to_most_original_1step'
                        ], help="cot is for chain of thought and standard is in context learning")

    parser.add_argument('--api_key', type=str, help="api key to be used for the experiment")
    parser.add_argument('--orgid', type=str, default=None)
    parser.add_argument('--api_base', type=str, help="api base url to be used for the experiment")
    parser.add_argument('--version', type=str)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--queries', type=int, default=500)
    parser.add_argument('--psplus', action='store_true')
    

    args = parser.parse_args()
    
    if args.psplus :
        args.zero_shot_prompt = "Let's first understand the problem, extract relevant variables and  their corresponding numerals, and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."

    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')

    filename=f"./logs/{args.model_name.replace('/', '|')}_{args.dataset}_{args.learning_mode}.log"
    output_file_name_raw = f"MODEL_{args.model_name.replace('/', '|')}DATASET_{args.dataset}METHOD_{args.learning_mode}.jsonl"
    output_file_name_results = f"MODEL_{args.model_name.replace('/', '|')}DATASET_{args.dataset}METHOD_{args.learning_mode}.txt"

    if args.psplus:
        output_file_name_raw = output_file_name_raw.replace(".jsonl", f"_ZERO_SHOT_PROMPT_psplus.jsonl")
        output_file_name_results = output_file_name_results.replace(".txt", f"_ZERO_SHOT_PROMPT_psplus.txt")
        filename = filename.replace(".log", f"_zero_shot_prompt_psplus.log")
    elif args.learning_mode == 'zero_shot_cot':
        output_file_name_raw = output_file_name_raw.replace(".jsonl", f"_ZERO_SHOT_PROMPT_{args.zero_shot_prompt}.jsonl")
        output_file_name_results = output_file_name_results.replace(".txt", f"_ZERO_SHOT_PROMPT_{args.zero_shot_prompt}.txt")
        filename = filename.replace(".log", f"_zero_shot_prompt_{args.zero_shot_prompt}.log")

    if args.version:
        output_file_name_raw = output_file_name_raw.replace(".jsonl", f"_VERSION_{args.version}.jsonl")
        output_file_name_results = output_file_name_results.replace(".txt", f"_VERSION_{args.version}.txt")
        filename = filename.replace(".log", f"_version_{args.version}.log")
        
    if args.options_later:
        output_file_name_raw = output_file_name_raw.replace(".jsonl", f"_options_later.jsonl")
        output_file_name_results = output_file_name_results.replace(".txt", f"_options_later.txt")
        filename = filename.replace(".log", f"_options_later.log")

    api_keys = []
    api_base = None

    if args.api_key is None:
        with open(os.path.join(ROOT_DIR, './openai_key.txt'), 'r') as f:
            for line in f.readlines():
                api_keys.append(line.strip())
        if(is_single_prompt_model(args)):
            api_keys = api_keys* 20
    else:
        api_keys += args.api_key.split(',')

    if args.api_base is not None:
        api_base = args.api_base

    dataset = args.dataset
    learning_mode = args.learning_mode
    model_name = args.model_name

    if os.path.isfile(f'prompts/{model_name}/{dataset}/{learning_mode}_prompt.txt'):
        prompt_path = f'prompts/{model_name}/{dataset}/{learning_mode}_prompt.txt'
    elif os.path.isfile(f'prompts/{dataset}/{learning_mode}_prompt.txt'):
        prompt_path = f'prompts/{dataset}/{learning_mode}_prompt.txt'
    elif os.path.isfile(f'prompts/{model_name}/{learning_mode}_prompt.txt'):
        prompt_path = f'prompts/{model_name}/{learning_mode}_prompt.txt'
    else:
        prompt_path = f'prompts/{learning_mode}_prompt.txt'

    all_test_questions = get_questions(args.dataset, args)

    if not args.sample:
        random.shuffle(all_test_questions)

    if args.sample:
        number_of_samples = args.queries

    else:
        number_of_samples = len(all_test_questions)
    level = logging.INFO
        
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(f'./saved_results/{output_file_name_raw}'):
        os.remove(f'./saved_results/{output_file_name_raw}')
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level)
    
    question_logger = logging.getLogger("logging_completed_questions")
    question_logger.setLevel(logging.INFO)
    question_logger_file_handler = logging.FileHandler(f'./saved_results/{output_file_name_raw}')
    question_logger_file_handler.setLevel(logging.INFO)
    question_logger_file_handler.setFormatter(logging.Formatter(None))
    question_logger.addHandler(question_logger_file_handler)
    
    if 'zero_shot' in args.learning_mode :
        prompt_instruction = ''
    else:
        with open(prompt_path, 'r') as input_file:
            prompt_instruction = input_file.read()

    answer_prompt = 'A:'
    if args.learning_mode=='zero_shot_cot' or args.learning_mode=='zero_shot_cot_augment':
        answer_prompt = f"A: {args.zero_shot_prompt}"
    elif args.learning_mode=='zero_shot' or args.learning_mode=='zero_shot_augment':
        answer_prompt = f"A: {args.zero_shot_prompt_stage2}"

    if 'codellama' in args.model_name and args.learning_mode == 'zero_shot_cot':
        templated_questions = add_instruction_prompt(all_test_questions, prompt_instruction, "Q: ", answer_prompt, args.learning_mode)
    else:
        templated_questions = add_prompt(all_test_questions, prompt_instruction, "Q: ", answer_prompt)

    logger.info("-------------------------")
    logger.info(args)
    print(args)
    logger.info("-------------------------")
    all_questions = []
    retry = 0
    prev_total = 0
    prev_correct = 0

    initial_batch_size = 20 * len(api_keys)

    if args.max_tokens:
        max_tokens = args.max_tokens
    else:
        if args.learning_mode.startswith('least_to_most') or '_rephrase' in args.learning_mode:
            max_tokens = 600
        else:
            max_tokens = 150

    if is_single_prompt_model(args):
        initial_batch_size = 1 * len(api_keys)

    print("max_tokens: ", max_tokens)

    print("len(api_keys): ", len(api_keys), ", initial_batch_size: ", initial_batch_size)

    print('prompt_file: ', prompt_path)
    print('log_file: ', filename)
    print('results_file: ', f'saved_results/{output_file_name_raw}')

    epochs = 1 # change this value for self-consistency experiments

    while epochs >0:
        print('-'*100)
        print('epoch:....', epochs)
        epochs-=1
        failures = templated_questions.copy()
        while True:
            _questions, failures, correct_count, total_count = run_queries(args, api_keys, api_base, failures, number_of_samples, max_tokens, prev_total, prev_correct, retry=retry, initial_batch_size=initial_batch_size )
            all_questions += _questions         
            if( len(failures)>0 and  retry <2 ): # can change this for questions that require longer answers
                max_tokens *=2
                retry+=1
                print("retry: ", retry)
                prev_total = total_count
                prev_correct = correct_count
            else:
                total_count += len(failures)
                for question_sample in failures:
                    question_logger.info(question_sample)
                break


    with open(f'./saved_results/{output_file_name_results}', 'w') as output_file:
        output_file.writelines(
            [f'correct_count:{correct_count}\n', f'total_count:{total_count}\n', f'accuracy:{correct_count/total_count}\n'])

    print("correct_count: ", correct_count)
    print("total_count: ", total_count)
    print("accuracy: ", correct_count/total_count)
