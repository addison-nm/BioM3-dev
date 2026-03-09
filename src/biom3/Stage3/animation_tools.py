import textwrap
from PIL import Image, ImageDraw, ImageFont
import imageio
import os

# convert the numerical labels tok characters
def convert_num_to_char(
        tokens: list,
        char_tokens: any
    ) -> str:
    return "".join([tokens[num] for num in char_tokens.tolist()])

# draw text onto white page
def draw_text(
        image: any,
        text: any,
        font: any,
        position: tuple=(0,0),
        max_width: any=None,
        fill: tuple=(0,0,0)
    ) -> None:

    draw = ImageDraw.Draw(image)
    if max_width:
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text
    draw.multiline_text(position, wrapped_text, font=font, fill=fill)


# create gif animation 
def generate_text_animation(
        text_list: list,
        text_animation_path: str,
        output_temp_path: str='./outputs/temp_files'
    ) -> None:

    # create images with text
    image_files = []
    for index, text in enumerate(text_list):
        
        img = Image.new('RGB', (600, 159), color=(255, 255, 255))  # Create a white image
        font = ImageFont.load_default()
        draw_text(img, text, font, position=(10, 10), max_width=80, fill=(0, 0, 0))

        # Save image to a temporary file
        os.makedirs(output_temp_path, exist_ok=True)
        # temp_file = f'./outputs/temp_image_{index}.png'
        temp_file = output_temp_path + f'/temp_image_{index}.png'
        img.save(temp_file)
        image_files.append(temp_file)

    # Read saved images and create a GIF
    images = [imageio.imread(file) for file in image_files]
    imageio.mimsave(
            text_animation_path,
            images,
            format='GIF',
            duration=0.2,
    )

    # clean up temp image files
    for file in image_files:
        os.remove(file)
    return 
