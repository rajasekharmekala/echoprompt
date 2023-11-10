import json
import re
import ast
from random import randint

def load_jsonlfile(filename):
    dt = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            try:
                question = json.loads(line)
            except:
                question = ast.literal_eval(line)
            dt[question.get('id', i)] = question
    return dt
def load_dictfile(filename):
    dt = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            question = ast.literal_eval(line)
            dt[question.get('id', i)] = question
    return dt

def get_correct_date(target_scores):
    for key, value in target_scores.items():
        if value == 1: return key

def get_questions(dataset, args={}):
    if dataset == 'gsm8k':
        test_path_questions = './datasets/gsm8k.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line) for line in input_file]

        for i,q in enumerate(current_qs):
            modified_question = q['question'].strip()
            golden_answer = q['answer'].split('####')[-1]
            all_test_questions.append({
                "question": modified_question,
                "answer": q["answer"],
                "id": i,
                "golden_answer": golden_answer,
                "perplexities": q.get("perplexities", {}),
                "rephrases":q['rephrases']
                })
    elif dataset == 'svamp':
        test_path_questions = './datasets/svamp.jsonl'

        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line) for line in input_file]
            
        for i,q in enumerate(current_qs):
            modified_question = q['question'].strip()
            golden_answer = q['answer']
            all_test_questions.append({
                "question": modified_question,
                "answer": q["answer"],
                "id": i,
                "golden_answer": golden_answer,
                "rephrases":q['rephrases'],
                "perplexities": q.get("perplexities", {})
                })
    elif dataset == 'multiarith' or dataset == 'singleeq' or dataset == 'add_sub' or dataset == 'singleop':
        test_path_questions = f'./datasets/{dataset}.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line) for line in input_file.readlines()]
        for i,q in enumerate(current_qs):
            modified_question = q['question'].strip()
            golden_answer = float(q['answer'])
            all_test_questions.append({
                "question": modified_question,
                "answer": golden_answer,
                "id": i,
                "golden_answer": golden_answer,
                "rephrases":q['rephrases'],
                "perplexities": q.get("perplexities", {})
                })
    elif dataset == 'drop_football' or dataset == 'drop_nonfootball':
        test_path_questions = f'./datasets/{dataset}.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file]

        for i,q in enumerate(current_qs):
            modified_question = q['question'].strip()
            golden_answer = q['golden_answer']
            all_test_questions.append({
                'question': modified_question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id': q['id'],
                'rephrases':[] if 'rephrases' not in q else q['rephrases']
        })
    elif dataset == 'drop_census' or dataset == 'drop_break':
        test_path_questions = f'./datasets/{dataset}.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file.readlines()]
        count = 0
        for i,q in enumerate(current_qs):
            modified_question = q['question'].strip()
            golden_answer = q['answer'][0][0]
            all_test_questions.append({
                'question': modified_question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id':count,
                'n_steps': q.get('n_steps', None),
                "perplexities": q.get("perplexities", {}),
                'rephrases': q['rephrases']})
            count+=1
    elif dataset == 'aqua':
        with open('./datasets/aqua.jsonl', 'r') as f:
            all_test_questions = [json.loads(line) for line in f]
        # (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
        def addOptions(options):
            return "Answer Choices: " + " ".join([f"({chr(97+i)}) { ')'.join(value.split(')')[1:] ) }" for i, value in enumerate(options)])

        for i, question in enumerate(all_test_questions):
            question['question'] = question['question']+ '\n' + addOptions(question["options"]) if not args.options_later else question['question']
            golden_answer = question['correct'].lower()
            question['golden_answer'] = golden_answer.strip()
            question['id'] = i 
            question["options"] = " ".join([f"({chr(97+i)}) { ')'.join(value.split(')')[1:] ) }" for i, value in enumerate(question["options"])])
            
    elif dataset == 'mathqa':
        with open('./datasets/mathqa.jsonl', 'r') as f:
            all_test_questions = [json.loads(line) for line in f]
        # (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
        def format_answer_choices(input_string):
            try:
                parts = input_string.split(',')
                values = []
                for part in parts:
                    choice, value = part.strip().split(') ')
                    values.append(value.strip())
                formatted_string = " ".join([f"({chr(97+i)}) {value}" for i, value in enumerate(values)])
            except:
                formatted_string = input_string
            return formatted_string
        
        def addOptions(options):
            return "Answer choices: "+ format_answer_choices(options)

        for i, question in enumerate(all_test_questions):
            question['question'] = question['Problem']+ '\n' + addOptions(question["options"]) if not args.options_later else question['Problem']
            golden_answer = question['correct'].lower()
            question['golden_answer'] = golden_answer.strip()
            question['id'] = i 
            question["options"] = format_answer_choices(question["options"])
            
    elif dataset == 'mmlu_ele' or dataset == 'mmlu_high':
        with open(f'./datasets/{dataset}.jsonl', 'r') as f:
            all_test_questions = [json.loads(line) for line in f]
        # (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
        def addOptions(options):
            return "Answer Choices: " + " ".join([f"({chr(97+i)}) {value}" for i, value in enumerate(options)])

        for i, question in enumerate(all_test_questions):
            question['question'] = question['question']+ '\n' + addOptions(question["choices"]) if not args.options_later else question['question']
            golden_answer = f"{chr(97+question['answer'])}".lower()
            question['golden_answer'] = golden_answer.strip()
            question['id'] = i 
            question['options'] = " ".join([f"({chr(97+i)}) {value}" for i, value in enumerate(question["choices"])])
    elif dataset == 'logiqa':
        with open('./datasets/logiqa.jsonl', 'r') as f:
            all_test_questions = [json.loads(line) for line in f]
        # (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
        def addOptions(options):
            return "Answer Choices: " + " ".join([f"({chr(97+i)}) {value}" for i, value in enumerate(options)])
        for i, question in enumerate(all_test_questions):
            question['question'] = question['context']+ " "+ question["query"]+'\n' + addOptions(question["options"])if not args.options_later else question['context']+ " "+ question["query"]
            golden_answer = f"{chr(97+question['correct_option'])}".lower()
            question['golden_answer'] = golden_answer.strip()
            question['id'] = i
    elif dataset == 'csqa':
        with open('./datasets/csqa.jsonl', 'r') as f:
            all_test_questions = [json.loads(line) for line in f]
        def addOptions(options):
            return "Answer Choices: " + " ".join([f"({chr(97+i)}) {value['text']}" for i, value in enumerate(options)])
        for i, question in enumerate(all_test_questions):
            question['original_data'] = question['question']
            question['options'] = question['original_data']['choices']
            question['question'] = question['original_data']['stem']+ '\n' + addOptions(question["options"])if not args.options_later else question['original_data']['stem']
            golden_answer = question['answerKey'].lower()
            question['golden_answer'] = '('+golden_answer.strip()+')'
            question['id'] = i
    elif dataset == 'strategyqa':
        test_path_questions = './datasets/strategyqa.json'
        with open(test_path_questions, 'r') as input_file:
            all_test_questions = json.loads(input_file.read())
        for i, question in enumerate(all_test_questions):
            golden_answer = 'yes' if question['answer'] else 'no'
            question['golden_answer'] = golden_answer
            question['question'] = question['facts'][0]+ ' '+ question['question']
            question['id'] = i
    elif dataset == 'coin_flip':
        test_path_questions = f'./datasets/{dataset}.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file]

        for i,q in enumerate(current_qs):
            question =  q['question'].strip()
            golden_answer = 'yes' if q['answer'] else 'no'
            if 'zero_shot' in args.learning_mode:
                question += ' Note that "flip" here means "reverse".'
            all_test_questions.append({
                'question': question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id': i,
                'rephrases':q.get('rephrases', [])
        })
    elif dataset == 'shuffled_objects':
        test_path_questions = f'./datasets/{dataset}.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file]

        for i,q in enumerate(current_qs):
            question =  q['question'].replace('\n\n', '').strip()
            question += '?'
            golden_answer = q["answer"].strip('.').lower()
            all_test_questions.append({
                'question': question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id': i,
                'rephrases':q.get('rephrases', [])
        })
    elif dataset == 'winogrande_xl':
        test_path_questions = f'./datasets/winogrande_xl.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file]

        for i,q in enumerate(current_qs):
            question =  q['sentence'].strip() + f""" In the previous sentence, does _ refer to "{q['option1']}" or "{q['option2']}"?"""
            golden_answer = q['option1'].lower() if q['answer'] =="1" else q['option2'].lower()
            all_test_questions.append({
                'question': question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id': i
        })
    elif dataset == 'date_understanding':
        test_path_questions = f'./datasets/date_understanding.jsonl'
        all_test_questions = []
        with open(test_path_questions, 'r') as input_file:
            current_qs = [json.loads(line)for line in input_file]
        for i,q in enumerate(current_qs):
            question =  q['question']
            golden_answer = q['answer']
            all_test_questions.append({
                'question': question,
                'answer': golden_answer,
                'golden_answer': golden_answer,
                'id': i,
                'rephrases':q['rephrases']
        })
    else:
        raise Exception("dataset not found")
    return all_test_questions

def extract_symbols(str, no_char=True):
    symbols =  re.findall(r"[-+]?(?:\d*[,\d+]*[\.:]{0,1}\d+)", str)
    return [remove_char(s) for s in symbols] if no_char else symbols

def remove_char(input_string):
    input_string = str(input_string)
    input_string = input_string.split('\n')[0]
    if "=" in input_string:
        input_string = input_string.split("=")[-1]
    if isinstance(input_string, str):
        for char in """!"$%'(),:;@[]^_`{|}~""":
            input_string = input_string.replace(char, '')
        try:
            return re.findall("[+|-]{0,1}\d+\.*\d*", input_string)[0]
        except:
            # print("input : ", input_string)
            return input_string
    return input_string

def is_equivalent (_exp, _golden_exp):
    for i in range(0, 100):
        while True:
            exp = _exp
            golden_exp = _golden_exp
            nums = {}
            percent = {}
            for i in range(1, 11):
                nums[f'NUM{i}'] = str(randint(1, 100))
            for i in range(1, 11):
                percent[f'PERCENT{i}'] = str(randint(1, 100))
            
            for key in nums.keys():
                golden_exp = golden_exp.replace(key, nums[key])
                exp = exp.replace(key, nums[key])
            for key in percent.keys():
                golden_exp = golden_exp.replace(key, percent[key])
                exp = exp.replace(key, percent[key])
            
            val = float(eval(golden_exp))

            if(val.is_integer()): break
        
        exp = exp.replace('$', '')
        exp = exp.replace('%', '')
        exp = exp.replace('X', '*')
        try:
            if float(eval(exp)) != float(eval(golden_exp)):
                # print(float(eval(exp)))
                # print(float(eval(golden_exp)))
                # print(nums)
                if 'PERCENT' in golden_exp:
                    print(exp+"--"+golden_exp+"--"+_exp+"--"+_golden_exp)
                return False
        except Exception as err:
            print('-------------EVAL_ERROR--------------', err)
            print(exp+"--"+golden_exp+"--"+_exp+"--"+_golden_exp)
            return False
    return True

def dates_equal(date1, date2):
    # Splitting the dates into month, day, and year components
    try:
        month1, day1, year1 = map(int, date1.split('/'))
        month2, day2, year2 = map(int, date2.split('/'))

        # Checking if all components are equal
        if month1 == month2 and day1 == day2 and year1 == year2:
            return True
    except:
        pass    
    return False
