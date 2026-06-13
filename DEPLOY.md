# Putting this online for your friends (plain-English guide)

Goal: a web link your ~7 friends can open on any phone or laptop — free, and
locked to just them. No coding required for these steps.

## What you'll use
**Streamlit Community Cloud** — free hosting that runs this app straight from
GitHub, with a built-in "only these emails can view it" setting. Perfect for a
small group.

## One-time setup (~10 minutes)

1. **Put the code on GitHub.**
   - Make a free GitHub account if you don't have one.
   - Create a new repository (private is fine) and upload this whole `worldcup2026`
     folder. *(I can do the git commands for you if you ask — just say the word.)*
   - Make sure `data/results.csv` is included — that's the brain of the model.

2. **Deploy on Streamlit.**
   - Go to **share.streamlit.io** and sign in with GitHub.
   - Click **"Create app"** → pick your repository.
   - Set **Main file path** to: `app/streamlit_app.py`
   - Click **Deploy**. In ~1 minute you'll get a link like
     `https://your-app.streamlit.app`.

3. **Lock it to your friends.**
   - In the app's **Settings → Sharing**, switch it to private and add your
     friends' email addresses (the ones they use to sign in to Google/GitHub).
   - Everyone else is blocked. That's the gate — no passwords to manage.

4. **(Optional) Add API keys for live results.**
   - In **Settings → Secrets**, paste:
     ```
     FOOTBALL_DATA_API_KEY = "your-key"
     API_FOOTBALL_KEY = "your-key"
     ```
   - Skip this and the app still works on the historical data — you just won't get
     brand-new fixtures auto-pulled.

## Day-to-day
- **Updating the app:** push a change to GitHub and it redeploys itself.
- **New match results during the tournament:** open the app and click
  **"🔄 Refresh latest results"** in the sidebar. It pulls new results and
  re-rates every team.

## Good to know
- The free tier is plenty for ~7 occasional users.
- Free hosts wipe their disk on restart — that's fine here: the app rebuilds its
  database from the committed CSV every time it boots, so nothing is lost.
