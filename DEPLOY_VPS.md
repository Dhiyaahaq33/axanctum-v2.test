# Axanctum v2 VPS Runbook

Minimal live test setup. Dashboard is optional and disabled when `DASHBOARD_URL` is empty.

## First install

```bash
git clone <GITHUB_REPO_URL> axanctum-v2
cd axanctum-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run foreground

```bash
export TELEGRAM_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
unset DASHBOARD_URL
python3 main.py
```

## Run with systemd

Create `/etc/systemd/system/axanctum-v2.service`:

```ini
[Unit]
Description=Axanctum v2 Telegram signal bot
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/axanctum-v2
Environment=TELEGRAM_TOKEN=...
Environment=TELEGRAM_CHAT_ID=...
Environment=DASHBOARD_URL=
ExecStart=/opt/axanctum-v2/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now axanctum-v2
sudo journalctl -u axanctum-v2 -f
```
