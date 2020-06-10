FROM devito_devito

COPY ./requirements.txt /app/requirements-daks.txt

RUN /venv/bin/python3 -m pip install --upgrade pip && /venv/bin/pip install --no-cache-dir -r /app/requirements-daks.txt