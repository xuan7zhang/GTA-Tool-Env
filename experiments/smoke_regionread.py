"""Smoke-test RegionRead through proxy #2 (16282). Waits for the tool server,
then calls RegionRead on a single-object task (164, white cat eyes) and a
multi-object task (9, dog in the middle) and prints the fused output plus the
raw TextToBbox for comparison."""
import io, json, os, sys, time, urllib.request
import requests
from PIL import Image

PROXY = 'http://127.0.0.1:16282'
TS = 'http://127.0.0.1:16182'
DS = os.path.join(os.environ['GTA_BIG'], 'data/gta_dataset')


def wait_ready(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(TS + '/openapi.json', timeout=5)
            if r.status_code == 200:
                paths = r.json().get('paths', {})
                if any('RegionRead' in p for p in paths):
                    print(f'[ready] tool server up at {int(time.time()-t0)}s; '
                          f'RegionRead exposed'); return True
        except Exception:
            pass
        time.sleep(10)
    print('[timeout] tool server not ready'); return False


def img_bytes(path):
    buf = io.BytesIO(); Image.open(path).convert('RGB').save(buf, format='PNG')
    buf.seek(0); return buf


def call(tool, path, **fields):
    data = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in fields.items()}
    r = requests.post(f'{PROXY}/{tool}', data=data,
                      files={'image': ('i.png', img_bytes(path), 'image/png')},
                      headers={'X-GTA-Task-Id': 'smoke'}, timeout=180)
    r.raise_for_status(); out = r.json()
    return out if isinstance(out, str) else json.dumps(out)


if not wait_ready():
    sys.exit(1)

cases = [
    ('164 single-object', 'image/image_357.jpg', 'white cat', 'eye color'),
    ('9  multi-object',    'image/image_28.jpg',  'dog',       'breed'),
]
for label, rel, text, attr in cases:
    path = os.path.join(DS, rel)
    print('\n' + '=' * 60 + f'\n[{label}]  find="{text}" attribute="{attr}"')
    try:
        raw = call('TextToBbox', path, text=text, top1=False)
        print('  TextToBbox(all):', raw[:200])
        rr = call('RegionRead', path, text=text, attribute=attr)
        print('  RegionRead ->', rr[:300])
    except Exception as e:
        print('  ERROR:', repr(e))
