FROM  devitocodes/devito:cpu-dev

COPY . daks/

RUN /venv/bin/python3 -m pip install --upgrade pip && /venv/bin/pip install --no-cache-dir -r /app/daks/requirements.txt

WORKDIR /app/daks/