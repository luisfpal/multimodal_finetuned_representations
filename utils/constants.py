"""Shared constants for VLM prompting: chat templates, system instructions, role dicts, and answer-choice lists."""

LLAVA_CHAT_TEMPLATE = """
{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}
""".strip()
GUIDE_TEXT = "\nAnswer ONLY with the option's letter from the given choices directly.\n"
ANSWER_TEXT = "\nAnswer:"
LLaVA1_5_SYSTEM_INSTRUCTION = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
INTERNVL2_SYSTEM_INSTRUCTION = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
INTERNVL3_5_SYSTEM_INSTRUCTION = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
LLaVA_OV1_5_SYSTEM_INSTRUCTION = "You are a helpful assistant."
CAMBRIAN_SYSTEM_INSTRUCTION = """
You are Cambrian, a highly intelligent multimodal AI trained by NYU Vision X. 
As a multimodal AI, you have the ability to process and analyze images. Whenever an image is present in the conversation, very carefully examine it and consider its content when formulating your response.
You should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions. 
"""
SYSTEM_ROLE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": LLaVA1_5_SYSTEM_INSTRUCTION,
        },
    ],
}
ASSISTANT_ROLE = {
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Answer:",
        },
    ],
}
SQA_ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
MMLU_ANSWER_CHOICES = ["A", "B", "C", "D"]
MMBENCH_ANSWER_CHOICES = ["A", "B", "C", "D"]
MME_ANSWER_CHOICES = ["Yes", "No"]
AI2D_ANSWER_CHOICES = ["A", "B", "C", "D"]
MMSTAR_ANSWER_CHOICES = ["A", "B", "C", "D"]
POPE_ANSWER_CHOICES = ["Yes", "No"]
COCOQA_VI_DIGITS_MAP = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
SHORT_CAPTION_PROMPT = (
    "Please look carefully at this image and provide a short, descriptive caption. "
    "Focus on the main objects, actions, and scene elements. "
    "Keep the caption concise but informative, similar to how you would describe "
    "the image to someone who cannot see it. "
    "Be specific about what you see without being overly detailed. "
    "THE CAPTION SHOULD BE A SINGLE SENTENCE, NOT MULTIPLE SENTENCES!!!"
)
