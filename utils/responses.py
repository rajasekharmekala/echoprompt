import time
from .api import get_responses, is_single_prompt_model
from sortedcollections import OrderedSet

def get_subquestions(question, answer):
    parts = answer.split(", we need to know: ")
    sub_q = OrderedSet()
    sub_questions = parts[1].split('\", \"')
    for _q in sub_questions:
        sub_q.add(_q.strip('.').strip('\"'))
    final_question = parts[0].split("To answer the question ")[-1].strip('\"')
    sub_q.add(final_question)
    return list(sub_q)

def get_final_response(data_batch, stage2_prompt, api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=10):
    for i, question in enumerate(data_batch):
        question['cur_batch_idx'] = i

    input_query = [question['question'] for question in data_batch]
    responses = get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args)

    
    for answer in responses:
        id = answer["index"]
        sub_q = get_subquestions(data_batch[id]['golden_question'], answer['text'])
        data_batch[id]['sub_q'] = answer['text']
        data_batch[id]['sub_q_list'] = sub_q
        data_batch[id]['sub_q_answers'] = []

    round = 1
    prev_batch = data_batch

    while True:
        next_query = []
        next_batch = []
        for i, answer in enumerate(responses):
            if(len(answer.keys()) ==0): continue
            id = answer["index"]
            question = prev_batch[id]
            answer["index"] = question['cur_batch_idx']
            question['latest_choices'] = answer
            sub_q = question['sub_q_list']
            sub_q_answers = question['sub_q_answers']
            if round>1:
                sub_q_answers.append(answer['text'].strip('\n'))
            if(len(question['sub_q_list']))>= round:
                x = stage2_prompt + "\nQ:"+ question['golden_question'] + "\nA: Let's break down this problem:" + "".join([f"""\nQuestion: {q}\nAnswer: {sub_q_answers[i].strip()}""" for i, q in enumerate(sub_q[:round-1])])+ f"\nQuestion: {sub_q[round-1]}\nAnswer:"
                question['sub_q'] = x
                next_query.append(x)
                next_batch.append(question)
        
        if(len(next_batch) ==0):
            break
        time.sleep(delay)
        responses = get_responses(next_query, api_keys, api_base, model_name, max_tokens, q_per_process, args)
        prev_batch = next_batch
        round+=1

    result = []
    for question in data_batch:
        if 'latest_choices' in question:
            result.append(question['latest_choices'])
    return result

def get_final_response_2step(data_batch, stage2_prompt, api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=10):
    input_query = [question['question'] for question in data_batch]
    responses = get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args)

    if(len(responses)==0):
        return responses
    for i, answer in enumerate(responses):
        full_answer = answer['text']
        id = answer["index"]
        golden_question = data_batch[id]['golden_question']
        sub_q = get_subquestions(golden_question, full_answer)
        data_batch[id]['sub_q'] = answer['text']
        data_batch[id]['n_steps'] = len(sub_q)
        # print(answer['text'])
        input_query[id] = stage2_prompt + "\nQ:"+ golden_question + "\nA: Let's break down this problem: " + " ".join([f"{i+1}. {q}" for i, q in enumerate(sub_q)])+ "\n"
    time.sleep(delay)
    return get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args)

def get_final_response_zero_shot(isCot, add_quote, data_batch, stage2_prompt, api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=10):
    input_query = [question['question'] for question in data_batch]
    
    stop = None
    if 'instruct' in model_name or is_single_prompt_model(model_name=model_name):
        stop = None
    else:
        stop = ["Q:", "Question:", "A:"]
    if add_quote:
        if stop ==None:
            stop = ["\""]
        else:
            stop.append("\"")
    responses = get_responses(input_query, api_keys, api_base, model_name, max_tokens if  isCot or 'codellama' in model_name else 20, q_per_process, args, stop_tokens=stop)

    if(len(responses)==0 or not isCot):
        for res in responses:
            res['finish_reason'] = "stop"
            if 'codellama' not in model_name:
                res['text'] = stage2_prompt + res['text']
        return responses
    
    append  = "\"" if add_quote else ""

    for i, answer in enumerate(responses):
        full_answer = answer['text'].replace("\n\n", " ")
        id = answer["index"]
        
        if args.options_later:
            data_batch[id]['sub_q'] = full_answer +  append + "\nThe available options are:\n"+ data_batch[id]["options"] + "\n"
            input_query[id] = input_query[id]+ full_answer +  append + "\nThe available options are:\n"+ data_batch[id]["options"] + "\n"+  stage2_prompt
        else:
            data_batch[id]['sub_q'] = full_answer
            # if 'codellama' in model_name:
            #     input_query[id] = input_query[id]+ full_answer +  append + f"""</s><s>[INST] {stage2_prompt} [/INST]"""
            # else:
            input_query[id] = input_query[id]+ full_answer +  append +  stage2_prompt
        # print('_'*20)
        # print(input_query[id])    
            
    time.sleep(delay)

    choices = get_responses(input_query, api_keys, api_base, model_name, 20, q_per_process, args)
    for choice in choices:
        choice['text'] = stage2_prompt + choice['text']
    return choices

def get_final_response_options_later(data_batch, stage2_prompt, api_keys, api_base, model_name, max_tokens, q_per_process, args, delay=10):
    input_query = [question['question'] for question in data_batch]
    responses = get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args, stop_tokens=["Q:", "A:", "|END|", "\n\n"])

    for i, answer in enumerate(responses):
        full_answer = answer['text'].replace("\n\n", " ")
        id = answer["index"]
        
        data_batch[id]['sub_q'] = full_answer +  "\nThe available options are:\n"+ data_batch[id]["options"] + "\n"
        input_query[id] = input_query[id]+ full_answer +  "\nThe available options are:\n"+ data_batch[id]["options"] + "\n"+  stage2_prompt
            
            
    time.sleep(delay)

    choices = get_responses(input_query, api_keys, api_base, model_name, 10, q_per_process, args)
    for choice in choices:
        choice['text'] = stage2_prompt + choice['text']
    return choices
