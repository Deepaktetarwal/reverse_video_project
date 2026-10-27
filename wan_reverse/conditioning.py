"""The task uses the ORIGINAL teacher prompt to request reverse-time video."""
MODE = 'original_prompt_reverse_video_v2'

def student_embedding(tensors,dropped=False):
    return tensors['negative_embeds' if dropped else 'forward_embeds']

def student_prompt(record):
    return record['prompt']
