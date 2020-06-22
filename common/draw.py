import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

font_dir = Path(__file__).parent.absolute()

def show_box(img_fn, boxes, labels, scores):
    original_image = Image.open(img_fn).convert('RGB')
    annotated_image = original_image
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype(str(font_dir/"calibril.ttf"), 15)
    
    for box, label, socre in zip(boxes, labels, scores):

        draw.rectangle(xy=box, outline=label)
        draw.rectangle(xy=[l + 1. for l in box], outline=label)  # a second rectangle at an offset of 1 pixel to increase line thickness
       
        text_size = font.getsize(str(label).upper())
        text_location = [box[0] + 2., box[1] - text_size[1]]
        textbox_location = [box[0], box[1] - text_size[1], box[0] + text_size[0] + 4., box[1]]
        draw.rectangle(xy=textbox_location, fill=label)
        draw.text(xy=text_location, text=str(label).upper(), fill='white', font=font)
    del draw
    return annotated_image
