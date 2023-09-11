import time
import multiprocessing
import numpy as np
import openai

DELAY = 20
# multiprocessing.set_start_method('spawn')

def get_open_ai_response(
        api_key, 
        api_base, 
        model_name,
        input_query,
        max_tokens,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Q:", "Question:", "\n\n",  "```"],
        delay=DELAY,
        max_retry=2,
        logprobs=None
        ):
    try:
        if(len(input_query) == 0): return {'choices': []}
        openai.api_key = api_key
        if api_base is not None:
            openai.api_base = api_base
        response = openai.Completion.create(
            # model=model_name,
            engine=model_name,
            prompt=input_query,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logprobs=logprobs,
            echo=True if logprobs !=None else False,
        )
        return response
    except openai.error.RateLimitError as err:
        print(err)
        max_retry -=1
        if max_retry==0:
            return {'choices': []}
        print(err)
        time.sleep(min([delay, 10]))
        return get_open_ai_response(api_key,
                                    api_base,
                                    model_name,
                                    input_query,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    top_p=top_p,
                                    frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty,
                                    stop=stop,
                                    max_retry = max_retry-1,
                                    delay=2*delay,
                                    logprobs=logprobs)


def get_open_ai_chat_response(
        api_key, 
        api_base, 
        model_name,
        input_query,
        max_tokens,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Q:", "Question:", "\n\n",  "```"],
        delay=DELAY,
        max_retry=2,
        logprobs=None
        ):
    try:
        # print('--------')
        # print(api_key, api_base, model_name, max_tokens)
        # print('--------')
        if(len(input_query) == 0): return {'choices': []}
        if(len(input_query) >1): raise Exception("only 1 prompt at a time")
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": input_query[0]}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        for choice in response['choices']:
            choice['text'] = choice['message']['content']
        return response
    except openai.error.RateLimitError as err:
        print(err)
        max_retry -=1
        if max_retry==0:
            return {'choices': []}
        print(err)
        time.sleep(delay)
        return get_open_ai_chat_response(api_key,
                                    api_base,
                                    model_name,
                                    input_query,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    top_p=top_p,
                                    frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty,
                                    stop=stop,
                                    max_retry = max_retry-1,
                                    delay=2*delay,
                                    logprobs=logprobs)




def runThread(api_key, api_base, model_name, input_query, max_tokens, queue, start_idx,stop_tokens, args):
    if (is_chat_completion(args)):
        r = get_open_ai_chat_response(api_key, api_base, model_name, input_query, max_tokens=max_tokens, logprobs=None, temperature=0.0, stop=stop_tokens)
    else:
        r = get_open_ai_response(api_key, api_base, model_name, input_query, max_tokens=max_tokens, logprobs=None, temperature=0.0, stop=stop_tokens)
    choices = r['choices']
    for choice in choices:
        choice['index'] = start_idx + choice['index']
    # print(choice)
    queue.put(choices)


def get_responses(input_query, api_keys, api_base, model_name, max_tokens, q_per_process, args, stop_tokens=["Q:", "Question:", "\n\n",  "```"]):
    responses = []
    queue = multiprocessing.Queue()
    processes = []

    if(len(api_keys)> len(input_query)): q_per_process=1
    for i, api_key in enumerate(api_keys):
        start_idx = i*q_per_process
        end_idx = (i+1)*q_per_process if len(api_keys)-1 != i else len(input_query)
        query = input_query[start_idx : end_idx]
        p = multiprocessing.Process(target=runThread, args=(api_key, api_base, model_name, query, max_tokens, queue, start_idx, stop_tokens, args))
        processes.append(p)
        p.start()

    liveprocs = list(processes)
    while liveprocs:
        try:
            while 1:
                result = queue.get(False, 0.01)
                responses += result
        except Exception as err:
            pass
        time.sleep(0.5)
        if not queue.empty():
            continue
        liveprocs = [p for p in liveprocs if p.is_alive()]
    return responses





def is_chat_completion(args={}, model_name=None):
    if model_name is None:
        model_name = args.model_name
    return model_name in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']