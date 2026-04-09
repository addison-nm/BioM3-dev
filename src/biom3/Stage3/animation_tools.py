import math
import textwrap
from dataclasses import dataclass
from typing import Callable
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
_PAD_IDX = 23
_MASK_COLOR = (50, 50, 60)
_SPECIAL_TOKENS = {'<START>', '<END>', '<PAD>'}
_START_COLOR = (60, 190, 190)
_END_COLOR = (210, 120, 100)
_PAD_COLOR = (0, 0, 0)
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

_BAR_H = 6
_LOGO_H = 28
_EXTRA_PAD = 2
_LOGO_TOP_K = 4

# ── Legend ────────────────────────────────────────────────────────────────

_LEGEND_LABELS = {
    "gauge": "Fill height = model confidence",
    "brightness": "Brightness = model confidence",
    "colorbar": "Bars = predicted AA probabilities",
    "logo": "Logo = predicted AA probabilities",
}

_LEGEND_SWATCHES = [
    ("Hydrophobic", (100, 130, 200)),
    ("Aromatic", (80, 110, 190)),
    ("Positive", (200, 80, 80)),
    ("Negative", (160, 80, 180)),
    ("Polar", (80, 180, 80)),
    ("Gly", (200, 160, 60)),
    ("Cys", (200, 200, 60)),
    ("Pro", (80, 160, 160)),
    ("Mask", _MASK_COLOR),
    ("START", _START_COLOR),
    ("END", _END_COLOR),
    ("PAD", _PAD_COLOR),
]

_SWATCH_SIZE = 10
_LEGEND_LINE_H = 14


def _draw_legend(draw, x0, y0, max_width, prob_style, font_sm):
    """Draw a legend strip with style description and color key swatches.

    Returns the total height consumed.
    """
    cy = y0

    label = _LEGEND_LABELS.get(prob_style, "")
    if label:
        draw.text((x0, cy), label, fill=_HEADER_COLOR, font=font_sm)
        cy += _LEGEND_LINE_H + 2

    cx = x0
    for name, color in _LEGEND_SWATCHES:
        text_w = int(font_sm.getlength(name))
        entry_w = _SWATCH_SIZE + 3 + text_w + 10
        if cx + entry_w > x0 + max_width and cx > x0:
            cx = x0
            cy += _LEGEND_LINE_H
        draw.rectangle([cx, cy, cx + _SWATCH_SIZE, cy + _SWATCH_SIZE],
                       fill=color, outline=_DIM_TEXT_COLOR)
        draw.text((cx + _SWATCH_SIZE + 3, cy - 1), name,
                  fill=_DIM_TEXT_COLOR, font=font_sm)
        cx += entry_w

    return cy + _LEGEND_LINE_H - y0


def _legend_height(max_width, font_sm, prob_style):
    """Pre-compute the pixel height the legend will occupy."""
    has_label = prob_style in _LEGEND_LABELS
    cy = (_LEGEND_LINE_H + 2) if has_label else 0
    cx = 0
    for name, _ in _LEGEND_SWATCHES:
        text_w = int(font_sm.getlength(name))
        entry_w = _SWATCH_SIZE + 3 + text_w + 10
        if cx + entry_w > max_width and cx > 0:
            cx = 0
            cy += _LEGEND_LINE_H
        cx += entry_w
    return cy + _LEGEND_LINE_H


# ── Metric annotation system ──────────────────────────────────────────────


@dataclass
class MetricAnnotation:
    """A per-position annotation rendered as a coloured box above or below
    each residue cell in the animation.

    Parameters
    ----------
    name : str
        Label for the metric (used in legends / logging).
    values : np.ndarray
        Scalar values in ``[0, 1]``.  Shape ``[steps, seq_len]`` for dynamic
        metrics that change each denoising step, or ``[seq_len]`` for static
        metrics that are constant across all frames (set ``static=True``).
    colormap : callable
        ``float → (R, G, B)`` mapping a value in ``[0, 1]`` to an RGB tuple.
    height : int
        Box height in pixels.
    position : str
        ``"above"`` or ``"below"`` the residue cell.
    static : bool
        If ``True``, *values* is 1-D ``[seq_len]`` and the same data is used
        for every frame.  If ``False`` (default), *values* must be 2-D
        ``[steps, seq_len]``.
    """
    name: str
    values: np.ndarray
    colormap: Callable[[float], tuple[int, int, int]]
    height: int = 6
    position: str = "above"
    static: bool = False

    def value_at(self, step: int, pos: int) -> float:
        if self.static:
            return float(self.values[pos])
        return float(self.values[step, pos])


# ── Built-in colormaps ────────────────────────────────────────────────────


def red_yellow_green(value):
    """Diverging colormap: red (0) → yellow (0.5) → green (1)."""
    if value < 0.5:
        t = value * 2
        return (220, int(60 + 140 * t), 60)
    t = (value - 0.5) * 2
    return (int(220 - 160 * t), 200, int(60 + 20 * t))


def blue_white_red(value):
    """Diverging colormap: blue (0) → white (0.5) → red (1)."""
    if value < 0.5:
        t = value * 2
        return (int(60 + 195 * t), int(60 + 195 * t), 255)
    t = (value - 0.5) * 2
    return (255, int(255 - 195 * t), int(255 - 195 * t))


# ── Convenience factories ─────────────────────────────────────────────────


def confidence_metric(probs, position="above", height=6):
    """Create a dynamic confidence MetricAnnotation from probability arrays.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``[steps, seq_len, num_classes]``.
    """
    return MetricAnnotation(
        name="confidence",
        values=probs.max(axis=-1),
        colormap=red_yellow_green,
        height=height,
        position=position,
    )


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
    if token_str == '<START>':
        return _START_COLOR
    if token_str == '<END>':
        return _END_COLOR
    if token_str == '<PAD>':
        return _PAD_COLOR
    return AA_COLORS.get(token_str, (140, 140, 140))


def _alpha_blend(fg, bg, alpha):
    """Blend fg color onto bg at given alpha (0 = fully bg, 1 = fully fg)."""
    return tuple(int(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg))


def _draw_logo(draw, x, y, width, height, probs_vec, tokens,
               font_logo=None, draw_letters=True):
    """Draw a stacked-bar sequence logo for one position.

    Bars are stacked bottom-up so the highest-probability AA is closest
    to the residue cell below.  When *draw_letters* is False, only the
    coloured bars are drawn (compact "colorbar" mode).
    """
    top_k = min(_LOGO_TOP_K, len(probs_vec))
    indices = np.argsort(probs_vec)[::-1][:top_k]

    cursor_y = y + height  # start at bottom, draw upward
    for idx in indices:
        bar_h = int(probs_vec[idx] * height)
        if bar_h < 1:
            continue
        bar_top = cursor_y - bar_h
        tok = tokens[idx]
        color = AA_COLORS.get(tok, (140, 140, 140))
        draw.rectangle([x, bar_top, x + width, cursor_y], fill=color)
        if draw_letters and font_logo is not None and bar_h >= 8:
            bbox = font_logo.getbbox(tok)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x + (width - tw) // 2
            ty = bar_top + (bar_h - th) // 2
            draw.text((tx, ty), tok, fill=_TEXT_COLOR, font=font_logo)
        cursor_y = bar_top


def _render_frame(token_indices, prev_indices, tokens, step, total_steps,
                  title, cols_per_row, font, font_sm, step_probs=None,
                  prob_style=None, font_logo=None, metrics=None):
    seq_len = len(token_indices)
    num_rows = math.ceil(seq_len / cols_per_row)
    actual_cols = min(seq_len, cols_per_row)

    metrics = metrics or []
    above_metrics = [m for m in metrics if m.position == "above"]
    below_metrics = [m for m in metrics if m.position == "below"]

    # Height contributions above the cell
    above_metrics_h = sum(m.height + _EXTRA_PAD for m in above_metrics)
    if prob_style == "colorbar":
        logo_h = _BAR_H + _EXTRA_PAD
    elif prob_style == "logo":
        logo_h = _LOGO_H + _EXTRA_PAD
    else:
        logo_h = 0
    above_total = above_metrics_h + logo_h

    # Height contributions below the cell
    below_total = sum(m.height + _EXTRA_PAD for m in below_metrics)

    row_stride = above_total + _CELL + below_total + _PAD

    margin = 12
    row_label_w = 40
    header_h = 50 if title else 32

    grid_x0 = margin + row_label_w
    grid_y0 = margin + header_h
    grid_w = actual_cols * _STRIDE
    grid_h = num_rows * row_stride

    img_w = grid_x0 + grid_w + margin
    legend_h = _legend_height(img_w - 2 * margin, font_sm, prob_style) + 8
    img_h = grid_y0 + grid_h + legend_h + margin

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
        row_y = grid_y0 + row * row_stride

        tok_str = tokens[token_indices[j]]
        is_aa = tok_str != '-' and tok_str not in _SPECIAL_TOKENS
        is_pad = tok_str == '<PAD>'

        # -- Above-cell annotations (metrics, then logo/colorbar) --
        cursor_y = row_y

        for m in above_metrics:
            if is_aa:
                v = m.value_at(step, j)
                draw.rectangle([x, cursor_y, x + _CELL, cursor_y + m.height],
                               fill=m.colormap(v))
            cursor_y += m.height + _EXTRA_PAD

        if step_probs is not None and is_aa and logo_h > 0:
            if prob_style == "colorbar":
                _draw_logo(draw, x, cursor_y, _CELL, _BAR_H,
                           step_probs[j], tokens, draw_letters=False)
            elif prob_style == "logo":
                _draw_logo(draw, x, cursor_y, _CELL, _LOGO_H,
                           step_probs[j], tokens,
                           font_logo=font_logo, draw_letters=True)
        cursor_y += logo_h

        # -- Residue cell --
        cy = cursor_y
        bg = _cell_color(tok_str)
        if step_probs is not None and is_aa:
            confidence = float(step_probs[j][token_indices[j]])
            faint_bg = _alpha_blend(bg, _BG_COLOR, 0.1)
            draw.rectangle([x, cy, x + _CELL, cy + _CELL], fill=faint_bg)
            fill_h = int(_CELL * confidence)
            if fill_h > 0:
                fill_top = cy + _CELL - fill_h
                draw.rectangle([x, fill_top, x + _CELL, cy + _CELL], fill=bg)
        elif step_probs is not None and is_pad:
            draw.rectangle([x, cy, x + _CELL, cy + _CELL], fill=_MASK_COLOR)
            pad_prob = float(step_probs[j][_PAD_IDX])
            fill_h = int(pad_prob * _CELL)
            if fill_h > 0:
                draw.rectangle(
                    [x, cy + _CELL - fill_h, x + _CELL, cy + _CELL],
                    fill=_PAD_COLOR,
                )
        else:
            draw.rectangle([x, cy, x + _CELL, cy + _CELL], fill=bg)

        if j in newly_unmasked:
            draw.rectangle([x, cy, x + _CELL, cy + _CELL],
                           outline=_HIGHLIGHT_COLOR, width=2)

        if tok_str == '-':
            char, color = '\u00b7', _DIM_TEXT_COLOR
        elif tok_str == '<PAD>':
            char, color = '\u00b7', _TEXT_COLOR
        elif tok_str in _SPECIAL_TOKENS:
            char, color = tok_str[1], _DIM_TEXT_COLOR   # S for START, E for END
        else:
            char, color = tok_str, _TEXT_COLOR

        bbox = font.getbbox(char)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + (_CELL - tw) // 2
        ty = cy + (_CELL - th) // 2 - 1
        draw.text((tx, ty), char, fill=color, font=font)

        # -- Below-cell annotations --
        cursor_y = cy + _CELL + _EXTRA_PAD
        for m in below_metrics:
            if is_aa:
                v = m.value_at(step, j)
                draw.rectangle([x, cursor_y, x + _CELL, cursor_y + m.height],
                               fill=m.colormap(v))
            cursor_y += m.height + _EXTRA_PAD

    # -- Row position labels (aligned to cell) --
    for row in range(num_rows):
        label = str(row * cols_per_row)
        ly = grid_y0 + row * row_stride + above_total + 3
        draw.text((margin, ly), label, fill=_DIM_TEXT_COLOR, font=font_sm)

    # -- Legend --
    legend_y0 = grid_y0 + grid_h + 4
    _draw_legend(draw, margin, legend_y0, img_w - 2 * margin,
                 prob_style, font_sm)

    return img


def generate_sequence_animation(frames, tokens, output_path,
                                probs=None, prob_style=None,
                                metrics=None, title=None, cols_per_row=50,
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
    probs : np.ndarray, optional
        Per-step conditional probabilities, shape ``[steps, seq_len, num_classes]``.
        When ``None``, renders without any confidence visualisation.
    prob_style : str, optional
        How to visualise the full probability distribution.  One of:

        - ``None`` or ``"brightness"`` — scale cell colour brightness by
          confidence (dim = uncertain, vivid = confident).
        - ``"colorbar"`` — draw compact stacked amino-acid bars above each
          cell (no letters).
        - ``"logo"`` — draw a sequence-logo (stacked bars with letters)
          above each cell, heights proportional to probability.

        Ignored when *probs* is ``None``.
    metrics : list[MetricAnnotation], optional
        Per-position scalar annotations rendered as coloured boxes above or
        below each residue cell.  Metrics with ``position="above"`` stack
        above the logo/colorbar; those with ``position="below"`` stack below
        the cell.  Both dynamic (per-step) and static (constant across
        frames) metrics are supported.
    title : str, optional
        Header text drawn on every frame.
    cols_per_row : int
        Sequence positions per row before wrapping.
    duration : float
        Seconds per frame.
    loop : int
        GIF loop count (0 = infinite).
    """
    if prob_style is None and probs is not None:
        prob_style = "brightness"

    font = _get_font(13)
    font_sm = _get_font(11)
    font_logo = _get_font(9) if prob_style == "logo" else None
    total_steps = len(frames)

    images = []
    prev = None
    for step, frame in enumerate(frames):
        step_probs = probs[step] if probs is not None else None
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
            step_probs=step_probs,
            prob_style=prob_style,
            font_logo=font_logo,
            metrics=metrics,
        )
        images.append(np.array(img))
        prev = frame

    # Hold final frame longer
    for _ in range(5):
        images.append(images[-1])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.mimsave(output_path, images, format='GIF', duration=duration, loop=loop)


# ── GIF-to-MP4 conversion ────────────────────────────────────────────────


def gif_to_mp4(gif_path, mp4_path=None, fps=None, codec="libx264",
               pixel_format="yuv420p"):
    """Convert a GIF file to MP4.

    Parameters
    ----------
    gif_path : str
        Path to the input GIF.
    mp4_path : str, optional
        Output path. Defaults to *gif_path* with ``.mp4`` extension.
    fps : float, optional
        Frames per second. If ``None``, inferred from GIF metadata
        (falls back to 10).
    codec : str
        FFmpeg video codec.
    pixel_format : str
        Pixel format for broad compatibility.

    Returns
    -------
    str
        The output MP4 path.
    """
    if mp4_path is None:
        mp4_path = os.path.splitext(gif_path)[0] + ".mp4"

    reader = imageio.get_reader(gif_path)
    meta = reader.get_meta_data()
    if fps is None:
        duration_ms = meta.get("duration", 100)
        fps = 1000.0 / max(duration_ms, 1)

    os.makedirs(os.path.dirname(mp4_path) or '.', exist_ok=True)
    try:
        writer = imageio.get_writer(mp4_path, fps=fps, codec=codec,
                                    pixelformat=pixel_format)
    except (ImportError, IOError) as exc:
        raise RuntimeError(
            "MP4 conversion requires ffmpeg. Install via: "
            "pip install imageio-ffmpeg"
        ) from exc

    for frame in reader:
        h, w = frame.shape[:2]
        frame = frame[:h - h % 2, :w - w % 2]
        writer.append_data(frame)

    writer.close()
    reader.close()
    return mp4_path


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
