# Setup & Submission Guide — Windows

**Who this is for:** you, on a Windows PC, opening this project for the first time with no prior Python background.

> **Mac/Linux users:** read `docs/setup.md` instead.

Follow the steps in order. Don't skip — each one depends on the ones before it.

> Anything in a grey code box (`like this`) is a command you type into **Windows Terminal** (or PowerShell) and press Enter. Open it from the Start menu: press the **Windows key**, type "Terminal" or "PowerShell", and hit Enter.

---

## Part 1 — Getting it running on your own PC

### Step 1. Check what you already have installed

Type these one at a time and press Enter. You're just checking what's there — it's fine if something is missing, we install it next.

```
python --version
git --version
```

You should see two version numbers. If either says "is not recognized", note it — Step 2 fixes it.

### Step 2. Install the things you don't have yet

Windows 10 and 11 come with a built-in installer called **winget**. We'll use it.

**If Python is missing** (or the version is below 3.11):

```
winget install Python.Python.3.11
```

Press `Y` if it asks to agree to terms. When finished, **close Windows Terminal and reopen it** so it picks up the new Python.

**If Git is missing:**

```
winget install Git.Git
```

Again, close and reopen Terminal afterwards.

**Optional but recommended — VS Code** (a free text editor that makes the placeholder-replacement step way easier):

```
winget install Microsoft.VisualStudioCode
```

### Step 3. Put the project on your PC and open it in Terminal

If a classmate sent you this project as a `.zip` file, unzip it somewhere convenient — say `C:\Users\YourName\Desktop\kwadwo-ml`. If you already cloned it from GitHub, use that folder.

In Terminal:

```
cd C:\Users\YourName\Desktop\kwadwo-ml
```

(Replace `YourName` and the path with wherever you actually put the project.)

If you type `dir` and press Enter, you'll see files like `app.py`, `rag`, `docs`, etc.

### Step 4. Allow PowerShell to run scripts (one-time)

By default, Windows PowerShell blocks local scripts — including the one that activates the Python sandbox. To allow it:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

When it asks, type `Y` and press Enter. You only do this once per user account.

### Step 5. Create a Python "virtual environment"

This is a fancy name for a private Python sandbox for this project so its packages don't mess with anything else. You only do this once.

```
python -m venv .venv
```

Nothing visible happens, but a hidden folder called `.venv` is created.

### Step 6. Activate the virtual environment

**Every time you open a new Terminal window** to work on this project, you must run this one line first:

```
.venv\Scripts\Activate.ps1
```

After this, you'll see `(.venv)` appear at the start of your terminal prompt. That means "you are inside the sandbox". If you forget, the next commands will fail.

If you get an error about execution policy, re-do Step 4.

### Step 7. Install the project's packages

```
python -m pip install -U pip
python -m pip install -r requirements.txt
```

This installs Streamlit, the AI library, and everything else. Takes **2–5 minutes** on first run (about 1 GB of downloads, mainly PyTorch).

Lots of text will scroll past. As long as it doesn't end in red "ERROR:" lines, you're fine.

### Step 8. Download the two datasets

The project needs Ghana's election results (a spreadsheet) and the 2025 Budget Statement (a PDF). Windows 10 and 11 have `curl` built in:

```
curl.exe -L -o data\Ghana_Election_Result.csv https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv
```

```
curl.exe -L -o data\2025-Budget-Statement-and-Economic-Policy_v4.pdf https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf
```

> **Why `curl.exe`?** In PowerShell, just `curl` is an alias for a different (and more awkward) command. Typing `curl.exe` forces the real curl program.

Check both files downloaded:

```
dir data
```

You should see both filenames listed.

### Step 9. Get a free Gemini API key

The chatbot uses Google's Gemini AI to generate answers. Free tier, no credit card:

1. In your browser, open **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account.
3. Click **"Create API key"**.
4. Pick "Create API key in new project" if prompted.
5. A long string like `AIzaSy...` appears. **Copy it** (click the copy icon).

### Step 10. Save the key into the project

```
copy .env.example .env
```

Now open the `.env` file in Notepad:

```
notepad .env
```

You'll see this line:

```
GEMINI_API_KEY=your_key_here
```

Replace `your_key_here` with the key you copied. **Do not add quotes around it.** Save (`Ctrl+S`) and close Notepad.

**Important:** never share this file or upload it to GitHub. The project's `.gitignore` already blocks `.env` from being uploaded.

### Step 11. Build the search index

The chatbot pre-processes the datasets into a searchable form. This happens once:

```
python scripts\build_index.py
```

You'll see progress messages: "Loading CSV... 615 rows", "Loading PDF... 252 pages", etc. Takes about **30 seconds** and prints "Done." when finished.

### Step 12. Run the chatbot

```
streamlit run app.py
```

A browser tab opens automatically at `http://localhost:8501`. The **first query takes about 30 seconds** (the AI model is loading). After that, every query is fast.

Try asking:
- *"What was the total education allocation in the 2025 budget?"*
- *"Which party won the 2020 election in Ashanti Region?"*
- *"Compare NPP performance in Volta Region with the 2025 budget priorities for that region."*

To stop the app, click back on your Terminal window and press `Ctrl+C`.

### Step 13. Run the automated tests (optional, but good)

```
pytest
```

You should see `19 passed`.

---

## Part 2 — Filling in your personal details

Several files contain placeholder text like `<STUDENT_NAME>`, `<INDEX_NUMBER>`, `<DEPLOYED_URL>`, `<USERNAME>`. You need to replace these with your real details before submitting.

**The easy way (recommended): use VS Code's global Find & Replace.**

1. Open the project folder in VS Code: in Terminal, type `code .` and press Enter (or open VS Code, `File → Open Folder`, pick the project folder).
2. Press **`Ctrl+Shift+F`** to open the search-across-files panel.
3. Click the **down-arrow icon** next to the search box to reveal the Replace box.
4. Click the **`.*` icon** (regex) — **off**. Click the **`Aa` icon** (case-sensitive) — **on**.
5. Search for: `<STUDENT_NAME>` → Replace with: your full name (e.g. `Kwadwo Asante`) → click **"Replace All"** (a little icon next to the replace field).
6. Repeat for `<INDEX_NUMBER>` → your index number (e.g. `ACITY10001234`).
7. Repeat for `<USERNAME>` → your GitHub username.
8. **Leave `<DEPLOYED_URL>` alone for now** — you'll fill that in after you deploy in Part 4.

After replacing, save all files: **`Ctrl+K S`** (hold Ctrl, tap K, release both, tap S).

**Verify:** In the VS Code search panel, search again for `<STUDENT_NAME>` — you should see "No results in workspace".

**The hard way (PowerShell, if you don't want VS Code):**

```powershell
# Replace <STUDENT_NAME> (put your name in the quotes)
Get-ChildItem -Recurse -Include *.py,*.md -Exclude .venv | ForEach-Object {
  (Get-Content $_.FullName) -replace '<STUDENT_NAME>', 'YOUR NAME HERE' | Set-Content $_.FullName
}
```

Then repeat with `<INDEX_NUMBER>` → your index, and `<USERNAME>` → your GitHub username.

---

## Part 3 — Upload to GitHub

### Step 1. Create the GitHub repository

1. Go to **https://github.com/new** in your browser.
2. **Repository name:** type `ai_` followed by your index number. For example, if your index is `ACITY10001234`, the repo name is `ai_ACITY10001234`. **This exact name is required by the exam.**
3. **Public** is fine.
4. Leave everything else unchecked — **do not** add a README, `.gitignore`, or license. Your project already has these.
5. Click **Create repository**.

GitHub shows you a page with commands. Ignore them — use ours below.

### Step 2. Make your project a git repository

Back in Terminal, inside the project folder:

```
git init
```

Tell Git who you are (use your GitHub email):

```
git config user.email "you@example.com"
git config user.name "Your Name"
```

### Step 3. Add and commit all the files

```
git add .
git commit -m "Initial commit: CS4241 RAG exam submission"
```

### Step 4. Connect to your GitHub repo and push

Replace `YOUR_GITHUB_USERNAME` and `YOUR_INDEX` below:

```
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ai_YOUR_INDEX.git
git push -u origin main
```

**GitHub will open a sign-in window in your browser.** Sign in. If it asks about "Git Credential Manager", approve it. Once authenticated, the push will complete.

Refresh the repo page on GitHub — you should see all your files.

> If the browser sign-in doesn't appear, you may need a **Personal Access Token** instead of a password. Go to **https://github.com/settings/tokens** → **Generate new token (classic)** → tick the `repo` box → **Generate**. Copy the token. When Git asks for a password, paste the token.

### Step 5. Add the lecturer as a collaborator

1. On your GitHub repo page, click **Settings** (top right).
2. Left sidebar → **Collaborators**.
3. Click **Add people**.
4. Type **`GodwinDansoAcity`** and select it.
5. Click **Add**.

**This step is non-negotiable** — the exam says "Failure to do so will result in getting nothing for your exams."

---

## Part 4 — Deploy to Streamlit Cloud

This puts your app on the internet so the lecturer can click a link and use it.

### Step 1. Sign in to Streamlit Cloud

1. Go to **https://share.streamlit.io**.
2. Click **"Continue with GitHub"**.
3. Approve the permissions it asks for.

### Step 2. Create the app

1. Click **"New app"** (top right).
2. Pick **"From existing repo"**.
3. **Repository:** pick `YOUR_USERNAME/ai_YOUR_INDEX`.
4. **Branch:** `main`.
5. **Main file path:** `app.py`.
6. Click **"Deploy"**.

First deploy takes **3–5 minutes** (installing everything fresh on their servers).

### Step 3. Add your Gemini key as a secret

While it's building (or after), click the **three dots (⋮)** next to your app's name → **Settings** → **Secrets**. A text box appears.

Paste this (replace `YOUR_KEY_HERE` with the actual key you used locally):

```
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

Click **Save**. The app restarts automatically.

### Step 4. Get your deployed URL

Once the build finishes, your URL appears at the top:

```
https://ai-yourindex.streamlit.app
```

Copy it.

### Step 5. Save the URL back into your project

In VS Code (Ctrl+Shift+F), search for `<DEPLOYED_URL>` and replace with your Streamlit URL. Save all files (`Ctrl+K S`).

Then commit and push again:

```
git add .
git commit -m "Add deployed URL to docs"
git push
```

### Step 6. Test the live app

Open your Streamlit URL in an **InPrivate / Incognito** browser window (to make sure you're not accidentally logged in). Ask it a question. It should work exactly like your local version.

---

## Part 5 — Run the adversarial evaluation

This produces the numbers for your experiment logs (4 marks).

```
python evaluation\run_eval.py
```

Takes about a minute. Creates `evaluation\results.json` with 6 rows (2 queries × 3 runs × 2 systems: RAG vs pure LLM).

Now **open `docs\experiment_logs.md`** in VS Code and fill in the tables by hand using what's in `results.json`. **Do not let an AI summarize it for you** — the exam explicitly says "Manual experiment logs (not AI-generated summaries)". Write in your own words.

---

## Part 6 — Record the 2-minute video

1. Open `docs\video_script.md` — that's your teleprompter.
2. Open your live Streamlit URL in a browser.
3. Record using the **Windows Game Bar**:
   - Press **`Win+G`** to open it.
   - Click the **circular "Record" button** in the Capture widget (top-left).
   - Go back to your browser, run your demo. Follow the script. Keep it under 2 minutes.
   - Press **`Win+Alt+R`** to stop.
   - Recordings save to `C:\Users\YourName\Videos\Captures` by default.
4. If Game Bar doesn't work, install **OBS Studio** (free, `winget install OBSProject.OBSStudio`) — a bit more setup but reliable.
5. Upload the video to Google Drive or YouTube (unlisted is fine) and get a shareable link.

---

## Part 7 — Submit by email

Open your email app. Compose a new message:

- **To:** `godwin.danso@acity.edu.gh`
- **Subject:** `CS4241-Introduction to Artificial Intelligence-2026:[YOUR_INDEX YOUR_NAME]`
  - Replace `YOUR_INDEX` and `YOUR_NAME` with your real values.
- **Body:**

```
Good day sir,

Below are the submission links for my CS4241 RAG assistant project.

GitHub repository: https://github.com/YOUR_GITHUB_USERNAME/ai_YOUR_INDEX
Deployed application: https://ai-yourindex.streamlit.app
Video walkthrough: <your Google Drive / YouTube link>

All documentation is in the `docs/` folder of the repo:
- README.md — overview and setup
- docs/architecture.md — system architecture with diagram
- docs/design_decisions.md — detailed reasoning for each design choice
- docs/experiment_logs.md — manual experiment logs
- docs/video_script.md — video script

I have added GodwinDansoAcity as a collaborator on the repository.

Thank you.

<YOUR NAME>
<YOUR INDEX NUMBER>
```

Send.

---

## Submission checklist (final sanity check before hitting Send)

- [ ] Local app runs (`streamlit run app.py`) and answers questions correctly.
- [ ] `pytest` prints `19 passed`.
- [ ] No `<STUDENT_NAME>`, `<INDEX_NUMBER>`, `<USERNAME>`, `<DEPLOYED_URL>` left in any file.
- [ ] GitHub repo exists with name `ai_<YOUR_INDEX>` exactly.
- [ ] `GodwinDansoAcity` added as GitHub collaborator.
- [ ] Streamlit Cloud app is live and responds to queries.
- [ ] `evaluation\results.json` exists and `docs\experiment_logs.md` has your hand-written observations filled in.
- [ ] Video recorded, under 2 minutes, uploaded to a shareable link.
- [ ] Email sent to `godwin.danso@acity.edu.gh` with the correct subject format and all 3 links.

---

## Troubleshooting

**"'python' is not recognized"**
Windows didn't pick up Python after install. Close ALL Terminal windows and reopen. If that doesn't fix it, try `py --version` instead of `python --version` — some Windows setups use `py`. If both fail, run `winget install Python.Python.3.11` again and reboot.

**"`.venv\Scripts\Activate.ps1` cannot be loaded because running scripts is disabled"**
Revisit Part 1 Step 4 — run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and try again.

**"ModuleNotFoundError: No module named 'streamlit'" (or similar)**
You forgot to activate the virtual environment. Run `.venv\Scripts\Activate.ps1` first. You'll see `(.venv)` at the start of your prompt when it worked.

**"curl : The term 'curl' is not recognized"**
You typed `curl` (PowerShell alias) instead of `curl.exe` (the real program). Use `curl.exe` exactly as written in Step 8.

**"GEMINI_API_KEY not set"**
Either the `.env` file doesn't exist (run `copy .env.example .env`), or you didn't paste your real key into it, or you wrapped the key in quotes (remove the quotes — the format is `GEMINI_API_KEY=AIzaSy...` with no spaces, no quotes).

**Deployed app errors about Gemini key but local works**
You didn't add the secret to Streamlit Cloud. Revisit Part 4 Step 3.

**`git push` asks repeatedly for credentials**
Either the browser sign-in window got closed, or you need a Personal Access Token. See the note at the end of Part 3 Step 4.

**The first query takes forever**
Normal — the AI model loads into memory on first query. Subsequent queries are fast. On Streamlit Cloud the first query after the app boots can take up to 60 seconds.

**Path errors with backslashes**
Windows paths use `\` but some tools expect `/`. In this guide, commands use `\` where they're Windows-native (`cd`, `copy`, `python scripts\build_index.py`) and `/` in the `curl.exe -o` argument which is cross-platform. If you see a "path not found" error, try the other slash direction.

**I accidentally committed my `.env` file with my key in it**
Revoke the key immediately at https://aistudio.google.com/app/apikey (delete it, make a new one). Then: `git rm --cached .env`, commit, push.
