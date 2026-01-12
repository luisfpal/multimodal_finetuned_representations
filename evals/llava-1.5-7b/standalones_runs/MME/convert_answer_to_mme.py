import os
import json
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment',
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args


def get_gt(data_path):
    GT = {}
    # Check for MME_Benchmark wrapper directory
    benchmark_root = os.path.join(data_path, "MME_Benchmark")
    if os.path.isdir(benchmark_root):
        data_path = benchmark_root
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT


def clean_answer_text(text):
    """
    Clean answer text according to readme.txt requirements:
    - Remove all newlines (replace with space)
    - Normalize whitespace (multiple spaces to single space)
    - Strip leading/trailing whitespace
    - Ensure answer is on a single line
    """
    if not text:
        return ""
    # Replace all newlines and carriage returns with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Normalize multiple spaces to single space
    text = ' '.join(text.split())
    # Strip leading/trailing whitespace
    return text.strip()

if __name__ == "__main__":

    args = get_args()

    GT = get_gt(
        data_path='MME_Benchmark_release_version/MME_Benchmark'
    )

    experiment = args.experiment

    result_dir = os.path.join('eval_tool', 'answers', experiment)
    os.makedirs(result_dir, exist_ok=True)

    answers = [json.loads(line) for line in open(os.path.join('answers', f'{experiment}.jsonl'))]

    results = defaultdict(list)
    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        results[category].append((file, answer['prompt'], answer['text']))
        
    for category, cate_tups in results.items():
        with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
            for file, prompt, answer in cate_tups:
                # Clean the answer text: remove newlines and normalize whitespace
                # According to readme.txt: "if your responses contain "\n", please delet it"
                cleaned_answer = clean_answer_text(answer)
                
                # Normalize prompt: remove guide text and ensure "Please answer yes or no." format
                # Remove guide text if present
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                
                # Normalize newlines in prompt to spaces
                prompt = prompt.replace('\n', ' ').strip()
                # Normalize multiple spaces to single space
                prompt = ' '.join(prompt.split())
                
                # Extract the question part (before any "Please answer" text)
                question_part = prompt
                if 'Please answer yes or no.' in question_part:
                    question_part = question_part.split('Please answer yes or no.')[0].strip()
                
                # Try to find matching GT entry
                gt_ans = None
                matching_prompt = None
                
                # First, try exact matches with different "Please answer" formats
                variations_to_try = [
                    question_part + ' Please answer yes or no.',
                    question_part + '  Please answer yes or no.',  # Double space
                ]
                
                # Also try the original prompt if it already has "Please answer"
                if 'Please answer yes or no.' in prompt:
                    variations_to_try.insert(0, prompt)
                
                # Try exact matches first
                for prompt_var in variations_to_try:
                    key = (category, file, prompt_var)
                    if key in GT:
                        gt_ans = GT[key]
                        matching_prompt = prompt_var
                        break
                
                # If not found, try matching with normalized spaces
                if gt_ans is None:
                    target_prompt = question_part + ' Please answer yes or no.'
                    normalized_target = ' '.join(target_prompt.split())
                    
                    for key in GT.keys():
                        if key[0] == category and key[1] == file:
                            normalized_gt = ' '.join(key[2].split())
                            if normalized_gt == normalized_target:
                                gt_ans = GT[key]
                                matching_prompt = key[2]  # Use the GT prompt format
                                break
                
                # If still not found, try matching just the question part
                if gt_ans is None:
                    normalized_question = ' '.join(question_part.split())
                    for key in GT.keys():
                        if key[0] == category and key[1] == file:
                            gt_question = key[2]
                            if 'Please answer yes or no.' in gt_question:
                                gt_question_part = gt_question.split('Please answer yes or no.')[0].strip()
                                normalized_gt_question = ' '.join(gt_question_part.split())
                                if normalized_gt_question == normalized_question:
                                    gt_ans = GT[key]
                                    matching_prompt = key[2]  # Use the GT prompt format
                                    break
                
                # If still not found, skip this entry with a warning
                if gt_ans is None:
                    print(f"Warning: Could not find GT for {category}/{file} with prompt: {prompt[:80]}...")
                    continue
                
                # Format output according to readme.txt:
                # Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
                tup = file, matching_prompt, gt_ans, cleaned_answer
                fp.write('\t'.join(tup) + '\n')
