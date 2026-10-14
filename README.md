# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and suggests improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF or DOCX**
- Optional **job description** for tailored keyword matching
- Overall ATS score (0-100) plus breakdown: keywords, formatting, experience, skills, education, readability
- Matched vs. missing keywords
- Formatting issues and prioritised improvement suggestions

## Run locally
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key at https://aistudio.google.com/apikey
4. Set the key (pick one):
   - Create `.streamlit/secrets.toml` containing:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - or set an environment variable `GEMINI_API_KEY`
   - or paste it into the app's sidebar
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | none (required) |
| `GEMINI_MODEL` | secrets / env var | `gemini-2.5-flash` |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (do **not** commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add `GEMINI_API_KEY = "your-key-here"`.
5. Click **Deploy**.

## Notes
- Scanned/image-only PDFs have no extractable text, which is also a real ATS problem. Use a text-based PDF or DOCX.
- The score is an AI estimate, not the output of any real ATS. Use it as guidance.
- Resumes are sent to Google's Gemini API. Don't upload anything you aren't comfortable sharing.
