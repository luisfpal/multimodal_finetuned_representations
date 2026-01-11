LLAVA_CHAT_TEMPLATE = """
{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}
""".strip()
GUIDE_TEXT = "\nAnswer ONLY with the option's letter from the given choices directly.\n"
ANSWER_TEXT = "\nAnswer:"
SYSTEM_INSTRUCTION = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
SYSTEM_ROLE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": SYSTEM_INSTRUCTION,
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
# COCO dataset info that at some point were useful to make the pycocoevalcap work
COCO_INFO = {
    "description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
    "url": "http://mscoco.org",
    "version": "1.0",
    "year": 2014,
    "contributor": "Microsoft COCO group",
    "date_created": "2015-01-27 09:11:52.357475",
}
COCO_LICENSES = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2,
        "name": "Attribution-NonCommercial License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
        "id": 3,
        "name": "Attribution-NonCommercial-NoDerivs License",
    },
    {
        "url": "http://creativecommons.org/licenses/by/2.0/",
        "id": 4,
        "name": "Attribution License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-sa/2.0/",
        "id": 5,
        "name": "Attribution-ShareAlike License",
    },
    {
        "url": "http://creativecommons.org/licenses/by-nd/2.0/",
        "id": 6,
        "name": "Attribution-NoDerivs License",
    },
    {
        "url": "http://flickr.com/commons/usage/",
        "id": 7,
        "name": "No known copyright restrictions",
    },
    {
        "url": "http://www.usa.gov/copyright.shtml",
        "id": 8,
        "name": "United States Government Work",
    },
]
