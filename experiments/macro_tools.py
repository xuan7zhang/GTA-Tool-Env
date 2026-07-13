"""Composition macro tools for the C-axis experiment. Each macro chains/combines
primitive tools by calling them THROUGH the proxy (so masks/probes still apply),
collapsing multiple orchestration steps into one tool call. CPU-only wrappers;
the real perception runs on the primitives.

Registered on the tool server via `--extra macro_tools.py PerceiveAll ...`.
"""
import io
import os
import re
import requests
from agentlego.types import Annotated, ImageIO, Info
from agentlego.tools.base import BaseTool

PROXY = os.getenv('GTA_PROXY_URL_INTERNAL', 'http://127.0.0.1:16281')


def _img_bytes(image):
    buf = io.BytesIO()
    image.to_pil().save(buf, format='PNG')
    buf.seek(0)
    return buf


def _img_post(tool, image, **params):
    r = requests.post(f'{PROXY}/{tool}', params=params,
                      files={'image': ('image.png', _img_bytes(image), 'image/png')},
                      headers={'X-GTA-Task-Id': os.getenv('GTA_MACRO_TASK', 'macro')},
                      timeout=180)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, str) else str(out)


def _form_post(tool, image, **fields):
    """Call a tool that takes an image plus scalar params. The agentlego server
    exposes scalars as FastAPI Form fields, so they must ride in the multipart
    body (data=), not the query string. Bools are lower-cased ('true'/'false')."""
    data = {k: (str(v).lower() if isinstance(v, bool) else str(v))
            for k, v in fields.items()}
    r = requests.post(f'{PROXY}/{tool}', data=data,
                      files={'image': ('image.png', _img_bytes(image), 'image/png')},
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


_BBOX_RE = re.compile(r'-?\d+(?:\.\d+)?')


class RegionRead(BaseTool):
    """Sequential-dependency composition = TextToBbox -> RegionAttributeDescription.
    Given an object description, LOCATE that object and DESCRIBE that region in one
    call. Hides the error-prone glue (parsing bbox coords out of the detector's
    '(x1,y1,x2,y2), score N' string and threading them back in) and returns only the
    region description -- so it shrinks output (pro-mask), unlike PerceiveAll."""

    default_desc = ('Locate an object by description and describe a chosen attribute '
                    'of that image region in one step. Give the object to find as '
                    '`text` and what to describe as `attribute` (e.g. "breed", '
                    '"color", "taste"). Use instead of calling TextToBbox and then '
                    'RegionAttributeDescription. Note: it picks the single highest-'
                    'confidence detection, so use the primitives directly when you '
                    'must choose among several objects by position (middle/left/'
                    'largest) or count them.')

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object to locate, described in English.')],
        attribute: Annotated[str, Info('The attribute of that object to describe, '
                                       'e.g. "breed", "color", "material".')],
    ) -> str:
        det = _form_post('TextToBbox', image, text=text, top1=True)
        nums = _BBOX_RE.findall(det)
        if len(nums) < 4:
            # detector returned 'No object found.' (or unpar's) -> report faithfully
            return f"Could not locate '{text}' in the image (detector said: {det})."
        x1, y1, x2, y2 = (int(round(float(n))) for n in nums[:4])
        bbox = f'({x1}, {y1}, {x2}, {y2})'
        desc = _form_post('RegionAttributeDescription', image, bbox=bbox, attribute=attribute)
        return f"{attribute} of '{text}' (region {bbox}): {desc}"
