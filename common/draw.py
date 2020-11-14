import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

font_dir = Path(__file__).parent.absolute()

c_dict = {1:'#e41a1c', 2:'#377eb8', 3:'#984ea3', 4:'#ff7f00', 5:'#f781bf'}


def show_box(img_fn, boxes, labels, scores):
    original_image = Image.open(img_fn).convert('RGB')
    annotated_image = original_image
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype(str(font_dir/"calibril.ttf"), 10)
    
    for box, label, score in zip(boxes, labels, scores):
        if not isinstance(box, list):
            box = list(box)
        color = c_dict[label]
        draw.rectangle(xy=box, outline=color) 
        draw.rectangle(xy=[l + 1. for l in box], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness

        text = '[%s] %.2f'%(label, score)
        text_size = font.getsize(text.upper())
        text_location = [box[0] + 2., box[1] - text_size[1]]
        textbox_location = [box[0], box[1] - text_size[1], box[0] + text_size[0] + 4., box[1]]
        draw.rectangle(xy=textbox_location, fill=color)
        draw.text(xy=text_location, text=text.upper(), fill='white', font=font)
    del draw
    return annotated_image
