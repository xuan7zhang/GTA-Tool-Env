"""Composition macro tools for the C-axis experiment. Each macro chains/combines
primitive tools by calling them THROUGH the proxy (so masks/probes still apply),
collapsing multiple orchestration steps into one tool call. CPU-only wrappers;
the real perception runs on the primitives.

Registered on the tool server via `--extra macro_tools.py PerceiveAll ...`.
"""
import io
import os
import requests
from agentlego.types import Annotated, ImageIO, Info
from agentlego.tools.base import BaseTool

PROXY = os.getenv('GTA_PROXY_URL_INTERNAL', 'http://127.0.0.1:16281')


def _img_post(tool, image, **params):
    buf = io.BytesIO()
    image.to_pil().save(buf, format='PNG')
    buf.seek(0)
    r = requests.post(f'{PROXY}/{tool}', params=params,
                      files={'image': ('image.png', buf, 'image/png')},
                      headers={'X-GTA-Task-Id': os.getenv('GTA_MACRO_TASK', 'macro')},
                      timeout=180)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, str) else str(out)


class PerceiveAll(BaseTool):
    """Comprehensively perceive an image in one step: returns BOTH a natural-language
    description of the scene AND all text recognized in the image. Use this instead
    of calling ImageDescription and OCR separately."""

    default_desc = ('Perceive an image in one step: returns both a description of '
                    'the image content and all recognized text. Use instead of '
                    'ImageDescription and OCR separately.')

    def apply(self, image: ImageIO) -> str:
        desc = _img_post('ImageDescription', image)
        text = _img_post('OCR', image)
        return f'Image description: {desc}\nRecognized text: {text}'
