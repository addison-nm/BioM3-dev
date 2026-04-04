import math
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import os


# ── Amino-acid color scheme (Clustal-inspired, grouped by physicochemical property) ──

AA_COLORS = {
    # Hydrophobic
    'A': (100, 130, 200), 'V': (100, 130, 200), 'I': (100, 130, 200),
    'L': (100, 130, 200), 'M': (100, 130, 200),
    'F': (80, 110, 190), 'W': (80, 110, 190),
    # Proline
    'P': (80, 160, 160),
    # Positive charge
    'K': (200, 80, 80), 'R': (200, 80, 80), 'H': (200, 120, 120),
    # Negative charge
    'D': (160, 80, 180), 'E': (160, 80, 180),
    # Polar
    'S': (80, 180, 80), 'T': (80, 180, 80),
    'N': (80, 180, 120), 'Q': (80, 180, 120),
    # Special residues
    'G': (200, 160, 60), 'C': (200, 200, 60), 'Y': (60, 180, 180),
    # Rare / ambiguous
    'X': (140, 140, 140), 'U': (140, 140, 140), 'Z': (140, 140, 140),
    'B': (140, 140, 140), 'O': (140, 140, 140),
}

_MASK_IDX = 0
_MASK_COLOR = (50, 50, 60)
_SPECIAL_TOKENS = {'<START>', '<END>', '<PAD>'}
_SPECIAL_COLOR = (40, 40, 48)
_BG_COLOR = (28, 28, 32)
_HIGHLIGHT_COLOR = (255, 215, 50)
_TEXT_COLOR = (235, 235, 235)
_DIM_TEXT_COLOR = (75, 75, 90)
_PROGRESS_BG = (48, 48, 58)
_PROGRESS_FG = (90, 175, 115)
_HEADER_COLOR = (190, 190, 200)

_CELL = 20
_PAD = 2
_STRIDE = _CELL + _PAD

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def _get_font(size):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _cell_color(token_str):
    if token_str == '-':
        return _MASK_COLOR
    if token_str in _SPECIAL_TOKENS:
        return _SPECIAL_COLOR
    return AA_COLORS.get(token_str, (140, 140, 140))


def _render_frame(token_indices, prev_indices, tokens, step, total_steps,
                  title, cols_per_row, font, font_sm):
    seq_len = len(token_indices)
    num_rows = math.ceil(seq_len / cols_per_row)
    actual_cols = min(seq_len, cols_per_row)

    margin = 12
    row_label_w = 40
    header_h = 50 if title else 32

    grid_x0 = margin + row_label_w
    grid_y0 = margin + header_h
    grid_w = actual_cols * _STRIDE
    grid_h = num_rows * _STRIDE

    img_w = grid_x0 + grid_w + margin
    img_h = grid_y0 + grid_h + margin

    img = Image.new('RGB', (img_w, img_h), _BG_COLOR)
    draw = ImageDraw.Draw(img)

    # -- Header --
    y = margin
    if title:
        draw.text((margin, y), title, fill=_HEADER_COLOR, font=font)
        y += 20

    # Step counter
    pct = (step + 1) / total_steps
    n_unmasked = sum(1 for j in range(seq_len) if token_indices[j] != _MASK_IDX)
    step_text = f"Step {step + 1}/{total_steps}  ({n_unmasked}/{seq_len} revealed)"
    draw.text((margin, y), step_text, fill=_HEADER_COLOR, font=font_sm)

    # Progress bar
    bar_x = margin + font_sm.getlength(step_text) + 14
    bar_y = y + 2
    bar_w = max(80, img_w - int(bar_x) - margin)
    bar_h = 10
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=_PROGRESS_BG)
    fill_w = int(bar_w * pct)
    if fill_w > 0:
        draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=_PROGRESS_FG)

    # -- Identify newly unmasked positions --
    newly_unmasked = set()
    if prev_indices is not None:
        for j in range(seq_len):
            if prev_indices[j] == _MASK_IDX and token_indices[j] != _MASK_IDX:
                newly_unmasked.add(j)

    # -- Draw grid --
    for j in range(seq_len):
        row = j // cols_per_row
        col = j % cols_per_row
        x = grid_x0 + col * _STRIDE
        cy = grid_y0 + row * _STRIDE

        tok_str = tokens[token_indices[j]]
        bg = _cell_color(tok_str)
        draw.rectangle([x, cy, x + _CELL, cy + _CELL], fill=bg)

        if j in newly_unmasked:
            draw.rectangle([x, cy, x + _CELL, cy + _CELL],
                           outline=_HIGHLIGHT_COLOR, width=2)

        # Display character
        if tok_str == '-':
            char, color = '\u00b7', _DIM_TEXT_COLOR
        elif tok_str in _SPECIAL_TOKENS:
            char, color = tok_str[1], _DIM_TEXT_COLOR   # S, E, P
        else:
            char, color = tok_str, _TEXT_COLOR

        bbox = font.getbbox(char)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + (_CELL - tw) // 2
        ty = cy + (_CELL - th) // 2 - 1
        draw.text((tx, ty), char, fill=color, font=font)

    # -- Row position labels --
    for row in range(num_rows):
        label = str(row * cols_per_row)
        ly = grid_y0 + row * _STRIDE + 3
        draw.text((margin, ly), label, fill=_DIM_TEXT_COLOR, font=font_sm)

    return img


def generate_sequence_animation(frames, tokens, output_path,
                                title=None, cols_per_row=50,
                                duration=0.15, loop=0):
    """Render a GIF showing the diffusion denoising process as a colored grid.

    Parameters
    ----------
    frames : list[np.ndarray]
        One 1-D integer array of token indices per diffusion step.
    tokens : list[str]
        Token vocabulary (index → string).
    output_path : str
        Path for the output ``.gif`` file.
    title : str, optional
        Header text drawn on every frame.
    cols_per_row : int
        Sequence positions per row before wrapping.
    duration : float
        Seconds per frame.
    loop : int
        GIF loop count (0 = infinite).
    """
    font = _get_font(13)
    font_sm = _get_font(11)
    total_steps = len(frames)

    images = []
    prev = None
    for step, frame in enumerate(frames):
        img = _render_frame(
            token_indices=frame,
            prev_indices=prev,
            tokens=tokens,
            step=step,
            total_steps=total_steps,
            title=title,
            cols_per_row=cols_per_row,
            font=font,
            font_sm=font_sm,
        )
        images.append(np.array(img))
        prev = frame

    # Hold final frame longer
    for _ in range(5):
        images.append(images[-1])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.mimsave(output_path, images, format='GIF', duration=duration, loop=loop)


# ── Legacy text-based animation (kept for backwards compatibility) ───────

def convert_num_to_char(tokens, char_tokens):
    return "".join([tokens[num] for num in char_tokens.tolist()])


def draw_text(image, text, font, position=(0, 0), max_width=None, fill=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    if max_width:
        wrapped_text = textwrap.fill(text, width=max_width)
    else:
        wrapped_text = text
    draw.multiline_text(position, wrapped_text, font=font, fill=fill)


def generate_text_animation(text_list, text_animation_path,
                            output_temp_path='./outputs/temp_files'):
    image_files = []
    for index, text in enumerate(text_list):
        img = Image.new('RGB', (600, 159), color=(255, 255, 255))
        font = ImageFont.load_default()
        draw_text(img, text, font, position=(10, 10), max_width=80, fill=(0, 0, 0))

        os.makedirs(output_temp_path, exist_ok=True)
        temp_file = output_temp_path + f'/temp_image_{index}.png'
        img.save(temp_file)
        image_files.append(temp_file)

    images = [imageio.imread(file) for file in image_files]
    imageio.mimsave(
        text_animation_path,
        images,
        format='GIF',
        duration=0.2,
    )

    for file in image_files:
        os.remove(file)
