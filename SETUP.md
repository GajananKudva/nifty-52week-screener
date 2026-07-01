# 52-Week High/Low Stock Screener — Setup & Operations Guide

## Architecture Overview

```
mailer.py          ← Orchestrator: runs the pipeline end-to-end
  └─ engine.py     ← Data engine: fetches, screens, values stocks
       ├─ yfinance          (price history + fundamentals)
       ├─ pandas / numpy    (calculations)
       └─ 2-Stage DCF       (intrinsic value per share)
  └─ NarrativeGenerator    ← LLM: OpenAI or Gemini
  └─ ReportMailer          ← Jinja2 render + SMTP delivery
       └─ template.html    ← HTML email template
```

---

## 1. Installation

### Step 1 — Clone / place files
Ensure all four files are in the same directory:
```
your-project/
  engine.py
  mailer.py
  template.html
  requirements.txt
  .env.example
```

### Step 2 — Create a Python virtual environment
```bash
# Create venv
python -m venv .venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If you only use OpenAI, you can skip `google-generativeai` (and vice versa).
> The app will gracefully fall back to a placeholder narrative if neither is installed.

---

## 2. Environment Variable Setup

```bash
# Copy the example file
cp .env.example .env
```

Then open `.env` in any text editor and fill in your values:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `perplexity`, `openai`, or `groq` |
| `PERPLEXITY_API_KEY` | If using Perplexity | From perplexity.ai/settings/api — Sonar grounds its own web search |
| `OPENAI_MODEL` | Yes | Deep-dive model. Perplexity: `sonar-pro` (or `sonar-reasoning-pro`, ~35% cheaper output). OpenAI: `gpt-4o-mini`. Groq: `llama-3.3-70b-versatile` |
| `QUICK_MODEL` | No | Fast pre-screen model. Perplexity default `sonar` |
| `PERPLEXITY_SEARCH_DOMAINS` | No | Comma-separated allowlist (max 20). Defaults to NSE/BSE/ET/Mint/MoneyControl/Reuters so retrieval stays on Indian small/mid-cap coverage |
| `PERPLEXITY_RECENCY` | No | `month` (default), `week`, `day`, or `hour` |
| `OPENAI_API_KEY` | If using OpenAI | From platform.openai.com |
| `GROQ_API_KEY` | If using Groq | From console.groq.com (free tier) |
| `SMTP_USER` | Yes (for email) | Your Gmail address |
| `SMTP_PASSWORD` | Yes (for email) | 16-char Gmail App Password (see below) |
| `TO_EMAILS` | Yes (for email) | Comma-separated recipient list |
| `TICKER_UNIVERSE` | No | Leave blank for built-in 70-stock universe |

### Gmail App Password Setup (required for Gmail SMTP)
1. Go to **myaccount.google.com** → Security
2. Enable **2-Step Verification** if not already on
3. Search for **App Passwords** in the search bar
4. Create a new App Password → select "Mail" and "Other"
5. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`) into `SMTP_PASSWORD`

> For corporate email, replace `SMTP_HOST` / `SMTP_PORT` with your IT-provided SMTP settings.

---

## 3. Running the Script

### Manual one-shot run (with email)
```bash
python mailer.py
```

### Render report only — no email sent (useful for testing)
```bash
python mailer.py --save-only
```

This saves a timestamped `report_YYYYMMDD_HHMM.html` to the current directory.
Open it in a browser to preview the report before enabling SMTP.

### Running on a subset for testing
Edit `TICKER_UNIVERSE` in `.env` to a small list:
```
TICKER_UNIVERSE=AAPL,MSFT,NVDA,JPM,XOM
```

---

## 4. Automated Scheduling (Cron)

### Linux / macOS — cron job
Run at 6:30 PM Eastern (23:30 UTC) Monday–Friday:

```bash
# Open crontab
crontab -e
```

Add this line (adjust paths to match your installation):
```cron
30 23 * * 1-5 /path/to/.venv/bin/python /path/to/mailer.py >> /path/to/screener.log 2>&1
```

#### Full example:
```cron
# Daily 52W screen — runs 6:30 PM ET (23:30 UTC) Mon-Fri
30 23 * * 1-5 /home/ubuntu/screener/.venv/bin/python /home/ubuntu/screener/mailer.py >> /home/ubuntu/screener/screener.log 2>&1
```

Verify your cron timezone:
```bash
timedatectl         # Linux
sudo systemsetup -gettimezone  # macOS
```

### Windows — Task Scheduler
1. Open **Task Scheduler** → Create Basic Task
2. Trigger: **Daily**, repeat Mon–Fri at `18:30` (6:30 PM)
3. Action: **Start a Program**
   - Program: `C:\path\to\.venv\Scripts\python.exe`
   - Arguments: `C:\path\to\mailer.py`
   - Start in: `C:\path\to\your-project\`

### Cloud (AWS / GCP / Azure)
Use a cron-based serverless trigger (AWS EventBridge → Lambda, GCP Cloud Scheduler → Cloud Run, etc.)
or a simple EC2/VM with the cron entry above.

---

## 5. Log Monitoring

All runs append to `screener.log` in the script directory:
```bash
# Tail live
tail -f screener.log

# View today's errors only
grep ERROR screener.log | grep $(date +%Y-%m-%d)

# Check last run summary
grep "Run complete\|Screen complete" screener.log | tail -5
```

---

## 6. Customisation Guide

### Change the ticker universe
Edit `DEFAULT_UNIVERSE` in `mailer.py`, or set `TICKER_UNIVERSE` in `.env`.

### Tighten / loosen the breakout band
```bash
BREAKOUT_THRESHOLD=0.01    # 1% — very tight (fewer, higher-quality signals)
BREAKOUT_THRESHOLD=0.05    # 5% — loose   (more signals)
```

### Raise the volume bar
```bash
VOLUME_SURGE_THRESH=2.0    # Only flag stocks with 2x+ normal volume
```

### Adjust the DCF model
```bash
DCF_DISCOUNT_RATE=0.12     # More conservative WACC (lower valuations)
DCF_TERMINAL_GROWTH=0.025  # 2.5% perpetuity growth
DCF_STAGE1_YEARS=7         # Longer explicit forecast period
```

### Switch LLM provider
```bash
LLM_PROVIDER=gemini        # Switch from OpenAI to Gemini
GEMINI_MODEL=gemini-1.5-pro  # Use the more capable model
```

---

## 7. File Reference

| File | Purpose |
|---|---|
| `engine.py` | All data fetching, technical screening, fundamentals, and DCF logic |
| `mailer.py` | Pipeline orchestrator, LLM narrative generator, SMTP email delivery |
| `template.html` | Jinja2 HTML email template (dark-theme, email-client safe) |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Environment variable template — copy to `.env` |
| `screener.log` | Auto-generated run log (appended each run) |
| `report_YYYYMMDD_HHMM.html` | Auto-generated local HTML report copies |

---

## 8. Troubleshooting

**`No module named 'yfinance'`** → Run `pip install -r requirements.txt` inside your venv.

**`SMTP Authentication Failed`** → For Gmail, use an App Password, not your account password.
Ensure 2-Step Verification is enabled on the Google account.

**`TemplateNotFound: template.html`** → Ensure `template.html` is in the same directory as `mailer.py`.

**`Empty price history for TICKER`** → The ticker may be delisted, incorrectly spelled, or not
available on Yahoo Finance. The error is logged and the script continues to the next ticker.

**`Negative/zero FCF — DCF skipped`** → Normal for companies with negative free cash flow
(early-stage, cyclical lows, etc.). The DCF column will show `N/A` for these names.

**`Rate limit` / `Too Many Requests` from yfinance** → Increase `RATE_LIMIT_SLEEP` to `0.75`
or reduce `RATE_LIMIT_BATCH` to `30` in `.env`.
