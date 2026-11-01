"""Local survey review and analysis server. No uploads are persisted."""
import csv
import hashlib
import io
import json
import math
import subprocess
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
MAX_BODY = 20 * 1024 * 1024


def ocr_executable():
    bundled = ROOT / 'tesseract' / 'tesseract.exe'
    executable = str(bundled) if bundled.is_file() else shutil.which('tesseract')
    if not executable:
        raise ValueError('이 PC에 OCR 엔진이 없습니다. 사진을 대조해 직접 입력하거나 CSV를 가져오세요. 통계 분석은 정상 사용할 수 있습니다.')
    return executable


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze(payload):
    if payload.get('confirmed') is not True:
        raise ValueError('응답 대조·검토 확인이 필요합니다.')
    raw = payload.get('csv', '')
    parsed = list(csv.reader(io.StringIO(raw)))
    if len(parsed) < 2 or not parsed[0]:
        raise ValueError('헤더와 응답 행이 있는 CSV가 필요합니다.')
    headers = [s.strip() for s in parsed[0]]
    if any(not s for s in headers) or len(set(headers)) != len(headers):
        raise ValueError('열 이름은 비어 있지 않고 중복되지 않아야 합니다.')
    if any(len(row) != len(headers) for row in parsed[1:]):
        raise ValueError('각 행의 열 개수가 헤더와 같아야 합니다.')
    frame = pd.DataFrame(parsed[1:], columns=headers)
    if len(frame) < 3:
        raise ValueError('분석에는 최소 3행이 필요합니다. 논문용 표본 적정성을 뜻하지 않습니다.')
    low, high = float(payload.get('low', 1)), float(payload.get('high', 5))
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError('척도 최솟값은 최댓값보다 작아야 합니다.')
    scales = payload.get('scales')
    if not isinstance(scales, list) or not scales:
        raise ValueError('분석할 척도를 하나 이상 정의하세요.')
    names, used, scores, summaries, warnings = set(), set(), {}, [], []
    for scale in scales:
        name = str(scale.get('name', '')).strip()
        items = scale.get('items', [])
        reverse = scale.get('reverse', [])
        if not name or name in names or not isinstance(items, list) or not items or len(set(items)) != len(items):
            raise ValueError('척도 이름과 문항 목록을 확인하세요. 중복은 허용하지 않습니다.')
        if any(x not in headers for x in items) or not set(reverse).issubset(items):
            raise ValueError(f'{name}: 열 이름 또는 역문항을 확인하세요.')
        if used.intersection(items):
            raise ValueError('동일 문항을 여러 척도에 중복 배정할 수 없습니다.')
        names.add(name)
        used.update(items)
        data = frame[items].copy()
        for item in items:
            source = data[item].str.strip()
            numeric = pd.to_numeric(source.replace('', np.nan), errors='coerce')
            invalid = source.ne('') & (numeric.isna() | ~np.isfinite(numeric) | (numeric < low) | (numeric > high) | (numeric % 1 != 0))
            if invalid.any():
                rows = [str(i + 2) for i in invalid[invalid].index[:8]]
                raise ValueError(f'{item}: 척도 범위 내 정수만 허용합니다. CSV 행 {", ".join(rows)}')
            data[item] = low + high - numeric if item in reverse else numeric
        complete = data.dropna()
        score = data.mean(axis=1).where(data.notna().all(axis=1))
        scores[name] = score
        n, k = len(complete), len(items)
        alpha = None
        if n >= 3 and k >= 2:
            total_var = complete.sum(axis=1).var(ddof=1)
            if total_var > 0:
                alpha = k / (k - 1) * (1 - complete.var(ddof=1).sum() / total_var)
        if alpha is None:
            warnings.append(f'{name}: 신뢰도 산출 불가(문항 수·유효 표본·분산 확인).')
        if n < 30:
            warnings.append(f'{name}: 유효 표본이 30명 미만입니다. 추정 안정성과 연구 설계를 검토하세요.')
        summaries.append(dict(name=name, items=items, reverse=reverse, n=n, excluded=len(frame)-n,
                              mean=score.mean(), sd=score.std(ddof=1), alpha=alpha,
                              missing_cells=int(data.isna().sum().sum())))
    correlations = []
    score_frame = pd.DataFrame(scores)
    columns = list(scores)
    for i, left in enumerate(columns):
        for right in columns[i+1:]:
            pair = score_frame[[left, right]].dropna()
            r = p = None
            if len(pair) >= 3 and all(pair[c].nunique() > 1 for c in pair):
                result = stats.pearsonr(pair[left], pair[right])
                r, p = result.statistic, result.pvalue
            correlations.append(dict(left=left, right=right, n=len(pair), r=r, p=p))
    return clean(dict(timestamp=datetime.now(timezone.utc).isoformat(), title=str(payload.get('title','설문 분석 리포트'))[:300],
                      sha256=hashlib.sha256(raw.encode()).hexdigest(), rows=len(frame), low=low, high=high,
                      summaries=summaries, correlations=correlations, warnings=warnings,
                      method='척도별 모든 문항이 관측된 응답자의 문항 평균을 사용. 결측 대체 없음. 신뢰도는 척도별 완전사례의 비표준화 Cronbach α(ddof=1). 상관은 척도점수 쌍별 완전사례의 양측 Pearson 검정. 다중검정 보정 없음.',
                      versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=__import__('scipy').__version__)))


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, content_type='application/json; charset=utf-8'):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health':
            self.send(200, '{"status":"ok"}')
        else:
            self.send(404, '{"error":"Not found"}')

    def do_POST(self):
        # Prevent cross-site requests against this local analysis service.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send(403, '{"error":"허용되지 않은 출처"}')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('업로드 크기는 20MB 이하여야 합니다.')
            body = self.rfile.read(length)
            if self.path == '/api/analyze':
                result = analyze(json.loads(body))
            elif self.path == '/api/ocr':
                from PIL import Image
                with tempfile.TemporaryDirectory() as folder:
                    target = Path(folder) / 'image.png'
                    with Image.open(io.BytesIO(body)) as img:
                        if img.width * img.height > 30_000_000:
                            raise ValueError('이미지는 3천만 픽셀 이하여야 합니다.')
                        img.convert('RGB').save(target)
                    run = subprocess.run([ocr_executable(), str(target), 'stdout', '-l', 'eng'], capture_output=True, timeout=45,
                                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                    if run.returncode:
                        raise ValueError('OCR 실행 실패. 사진을 확인하세요.')
                    result = dict(text=run.stdout.decode('utf-8', errors='replace'),
                                  note='영문·숫자 OCR 참고값입니다. 체크박스·필기·한국어 자동 판독과 응답 매핑은 지원하지 않습니다. 원본 대조 후 직접 입력하세요.')
            else:
                return self.send(404, '{"error":"Not found"}')
            self.send(200, json.dumps(result, ensure_ascii=False, allow_nan=False))
        except (ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired, OSError) as error:
            self.send(400, json.dumps(dict(error=str(error)), ensure_ascii=False))


def main():
    import argparse
    import threading
    import webbrowser
    parser = argparse.ArgumentParser()
    frozen = getattr(sys, 'frozen', False)
    parser.add_argument('--port', type=int, default=0 if frozen else 8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        url = f'http://{args.host}:{server.server_port}'
        print(f'Survey Studio: {url}', flush=True)
        print('Keep this window open while using the app. Close it to stop.', flush=True)
        if frozen and not args.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
