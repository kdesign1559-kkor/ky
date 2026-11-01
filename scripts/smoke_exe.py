"""Run on Windows to exercise the actual packaged executable without Python imports."""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

exe = sys.argv[1]
port = 18743
process = subprocess.Popen([exe, '--no-browser', '--port', str(port)])
base = f'http://127.0.0.1:{port}'
try:
    for attempt in range(120):
        if process.poll() is not None:
            raise RuntimeError(f'EXE exited early: {process.returncode}')
        try:
            if json.load(urllib.request.urlopen(base + '/health', timeout=1))['status'] == 'ok':
                break
        except (OSError, urllib.error.URLError):
            time.sleep(.5)
    else:
        raise RuntimeError('Packaged EXE did not become ready')
    assert '논문 분석실' in urllib.request.urlopen(base).read().decode('utf-8')
    payload = dict(csv='a,b\n1,1\n2,2\n3,3\n', confirmed=True,
                   scales=[dict(name='test', items=['a', 'b'], reverse=[])])
    request = urllib.request.Request(base + '/api/analyze', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(request))
    assert result['summaries'][0]['n'] == 3
    assert abs(result['summaries'][0]['alpha'] - 1) < 1e-10
    print('Packaged EXE: health, embedded HTML, actual statistics passed')
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
