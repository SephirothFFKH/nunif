@rem Install python 3.10
py -3.12 -m venv venv
call .\venv\Scripts\activate
pip install --upgrade --no-cache-dir -r requirements-torch.txt
pip install --upgrade --no-cache-dir -r requirements.txt
pip install --upgrade --no-cache-dir -r requirements-gui.txt
if exist "av-13.1.0-cp312-cp312-win_amd64.whl" (
  pip install --upgrade --no-cache-dir -r requirements-pyav.txt
)
python -m waifu2x.download_models
python -m waifu2x.web.webgen
python -m iw3.download_models
pause
