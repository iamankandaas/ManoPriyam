# app/routes.py

# --- FINAL IMPORTS ---
import openai
import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, session, jsonify, current_app
from urllib.parse import urlparse
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import (
    User,
    Entry,
    Badge,
    Quest,
    QuestFrequency,
    QuestCompletion,
)
from app.emotion import get_text_emotion, get_face_emotion
from datetime import datetime, date, timedelta
from collections import Counter
from sqlalchemy import func
import json
from wordcloud import WordCloud, STOPWORDS
from zoneinfo import ZoneInfo
import statistics
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# load once, at module import
embedder = SentenceTransformer('all-MiniLM-L6-v2')

bp = Blueprint('bp', __name__)

# --- NEW CONTEXT-AWARE SUGGESTION FUNCTION ---
def get_openai_suggestion(journal_text: str, user_preferences: str):
    """Generates a highly personalized and context-aware suggestion."""
    try:
        # The python-dotenv in your config should load the key automatically
        if not os.environ.get('OPENAI_API_KEY'):
            print("WARNING: OPENAI_API_KEY not found. Skipping suggestion.")
            return "Suggestion feature is currently unavailable."

        # The final, intelligent prompt
        prompt = f"""
        A user has written the following journal entry:
        ---
        "{journal_text}"
        ---
        The user's interests are: {user_preferences}.

        Your Task:
        Write a short, empathetic, and human-like reply (2-3 sentences).
        1. Acknowledge the specifics of their entry (both the good and bad parts).
        2. Validate their feelings and their personal progress.
        3. If it feels natural, subtly suggest how ONE of their interests could relate to their situation. Do not just list their hobbies.
        """
            
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a wise and empathetic journaling assistant who replies to entries like a thoughtful friend."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "Sorry, I couldn't think of a reply right now. Please check back later."




@bp.route('/')
def landing():
    return render_template('landing.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('bp.entry'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user is None or not user.check_password(password):
            flash('Invalid email and/or password', 'danger')
            return redirect(url_for('bp.login'))
        login_user(user)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('bp.entry')
        return redirect(next_page)
    return render_template('login.html')

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('bp.entry'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        preferences = request.form.get('preferences')
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'warning')
            return redirect(url_for('bp.register'))
        user = User(email=email, preferences=preferences)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('bp.login'))
    return render_template('register.html')

@bp.route('/logout')
@login_required
def logout():
    # clear any pending flash messages so they don’t carry over after logout
    session.pop('_flashes', None)
    logout_user()
    return redirect(url_for('bp.landing'))

@bp.route('/entry', methods=['GET', 'POST'])
@login_required
def entry():
    if request.method == 'POST':
        text = request.form['text_input']
        image_data = request.form.get('image_data', '')

        # run detectors
        text_label, text_conf = get_text_emotion(text)
        face_label, face_conf = get_face_emotion(image_data) if image_data else (None, 0.0)

        # combine with weights
        weights = {'text': 0.7, 'face': 0.3}
        scores = {text_label: text_conf * weights['text']}
        if face_label:
            scores[face_label] = scores.get(face_label, 0.0) + face_conf * weights['face']
        combined_label = max(scores, key=scores.get)
        combined_conf = scores[combined_label]

        # --- NEW: Call OpenAI for a personalized suggestion ---
        ai_suggestion = get_openai_suggestion(text, current_user.preferences)

        # persist entry
        e = Entry(
            user_id=current_user.id,
            text_input=text,
            text_emotion=text_label,
            text_confidence=text_conf,
            face_emotion=face_label,
            face_confidence=face_conf,
            combined_emotion=combined_label,
            combined_confidence=combined_conf,
            suggestion=ai_suggestion  # <-- Save the new suggestion to the DB
        )
        db.session.add(e)

        # ── XP & LEVEL UP ─────────────────────────────────────────
        XP_PER_ENTRY = 10
        new_lvl = current_user.add_xp(XP_PER_ENTRY)
        if new_lvl:
            flash(f"✨ You reached Level {new_lvl}! (+{XP_PER_ENTRY} XP)", 'success')
        else:
            flash(f"+{XP_PER_ENTRY} XP earned", 'info')

        # ── STREAK LOGIC ────────────────────────────────────────────
        today = date.today()
        last = current_user.last_entry_date
        if last == today:
            pass
        elif last == today - timedelta(days=1):
            current_user.current_streak += 1
        else:
            current_user.current_streak = 1
        current_user.last_entry_date = today

        # ── BADGE LOGIC ─────────────────────────────────────────────
        new_streak = current_user.current_streak
        badge = Badge.query.filter_by(threshold=new_streak).first()
        if badge and badge not in current_user.badges:
            current_user.badges.append(badge)
            flash(f'🎉 You earned the “{badge.name}” badge!', 'success')

        # ── QUEST COMPLETION ──────────────────────────────────────────
        today = date.today()

        # 1) Daily Check‑In
        daily_quest = Quest.query.filter_by(frequency=QuestFrequency.DAILY).first()
        if daily_quest:
            already_done = QuestCompletion.query.filter_by(
                user_id=current_user.id,
                quest_id=daily_quest.id,
                completed_on=today
            ).first()
            if not already_done:
                # award the quest
                qc = QuestCompletion(
                    user_id    = current_user.id,
                    quest_id   = daily_quest.id,
                    completed_on = today
                )
                db.session.add(qc)
                # award XP
                current_user.add_xp(daily_quest.xp_reward)
                flash(f"✅ Quest completed: {daily_quest.name}! (+{daily_quest.xp_reward} XP)", 'success')

        # 2) Weekly Streak‑Keeper
        weekly_quest = Quest.query.filter_by(frequency=QuestFrequency.WEEKLY).first()
        if weekly_quest and current_user.current_streak >= 7:
            # mark it once per calendar week (use ISO week)
            year, week_num, _ = today.isocalendar()
            already_done = QuestCompletion.query.filter_by(
                user_id    = current_user.id,
                quest_id   = weekly_quest.id,
                week_year  = year,
                week_number= week_num
            ).first()
            if not already_done:
                qc = QuestCompletion(
                    user_id     = current_user.id,
                    quest_id    = weekly_quest.id,
                    completed_on = today,
                    week_year    = year,
                    week_number  = week_num
                )
            db.session.add(qc)
            current_user.add_xp(weekly_quest.xp_reward)
            flash(f"🏅 Quest completed: {weekly_quest.name}! (+{weekly_quest.xp_reward} XP)", 'success')

        # 3) Consistent Trio: 3-day streak
        ct = Quest.query.filter_by(name="Consistent Trio").first()
        if ct and current_user.current_streak >= 3:
            week_start = today - timedelta(days=today.weekday())
            done = QuestCompletion.query.filter_by(
                user_id      = current_user.id,
                quest_id     = ct.id,
                completed_on = week_start
            ).first()
            if not done:
                db.session.add(QuestCompletion(
                    user_id      = current_user.id,
                    quest_id     = ct.id,
                    completed_on = week_start
                ))
                current_user.add_xp(ct.xp_reward)
                flash(f"🏃 Quest complete: Consistent Trio! (+{ct.xp_reward} XP)", 'success')

        # 4) Level 2 Achiever: leveled up to 2
        l2 = Quest.query.filter_by(name="Level 2 Achiever").first()
        # we still have `new_lvl` from XP logic above
        if l2 and new_lvl == 2:
        # use today as completed_on
            done = QuestCompletion.query.filter_by(
                user_id      = current_user.id,
                quest_id     = l2.id,
                completed_on = today
            ).first()
            if not done:
                db.session.add(QuestCompletion(
                    user_id      = current_user.id,
                    quest_id     = l2.id,
                    completed_on = today
                ))
                # they already got XP via add_xp → no need to re-award
                flash(f"🎯 Quest complete: Level 2 Achiever!", 'success')


        db.session.commit()

        return render_template(
            'entry.html',
            result=True,
            text_emotion=text_label,
            text_confidence=round(text_conf * 100, 1),
            face_emotion=face_label,
            face_confidence=round(face_conf * 100, 1),
            combined_emotion=combined_label,
            combined_confidence=round(combined_conf * 100, 1),
            xp=current_user.xp,
            level=current_user.level,
            suggestion=ai_suggestion # <-- Pass suggestion to the template to be displayed
        )
    return render_template('entry.html')

@bp.route('/history')
@login_required
def history():
    q = Entry.query.filter_by(user_id=current_user.id)

    # filters
    start = request.args.get('start_date')
    end   = request.args.get('end_date')
    month = request.args.get('month', type=int)
    year  = request.args.get('year', type=int)
    comb  = request.args.get('combined_emotion')
    fb    = request.args.get('feedback')

    if start:
        dt = datetime.fromisoformat(start)
        q = q.filter(Entry.timestamp >= dt)
    if end:
        dt2 = datetime.fromisoformat(end).replace(hour=23, minute=59, second=59)
        q = q.filter(Entry.timestamp <= dt2)
    if month and year:
        q = q.filter(
            db.extract('month', Entry.timestamp) == month,
            db.extract('year',  Entry.timestamp) == year
        )
    if comb:
        q = q.filter(Entry.combined_emotion == comb)
    if fb == 'liked':
        q = q.filter(Entry.feedback.is_(True))
    elif fb == 'disliked':
        q = q.filter(Entry.feedback.is_(False))

    entries = q.order_by(Entry.timestamp.desc()).all()
    all_emotions = [row[0] for row in db.session.query(Entry.combined_emotion).distinct()]

    return render_template(
        'history.html',
        entries=entries,
        start_date=start or '',
        end_date=end or '',
        month=month or '',
        year=year or '',
        combined_emotion=comb or '',
        feedback=fb or 'all',
        all_emotions=all_emotions
    )

@bp.route('/feedback/<int:entry_id>', methods=['POST'])
@login_required
def feedback(entry_id):
    e = Entry.query.get_or_404(entry_id)
    if e.user_id != current_user.id:
        flash('Not authorized to give feedback on this entry', 'danger')
        return redirect(url_for('bp.history'))
    fb = request.form.get('feedback')
    e.feedback = (fb == 'up')
    db.session.commit()

# ── FEEDBACK-QUEST LOGIC ────────────────────────────
    # only count entries with feedback set (either 👍 or 👎)
    # over the current ISO-week
    from datetime import date, timedelta
    today = date.today()
    week_start = today - timedelta(days=today.weekday())

    # how many times this user has given feedback this week?
    fed_count = (
      Entry.query
           .filter(
             Entry.user_id==current_user.id,
             Entry.feedback.isnot(None),
             Entry.timestamp >= week_start
           )
           .count()
    )

    # find our “Feedback Champion” quest
    fbq = Quest.query.filter_by(name="Feedback Champion").first()
    if fbq and fed_count >= 5:
        # check if already completed this week
        exists = QuestCompletion.query.filter_by(
            user_id     = current_user.id,
            quest_id    = fbq.id,
            completed_on=week_start
        ).first()
        if not exists:
            # award completion + XP
            db.session.add(QuestCompletion(
                user_id      = current_user.id,
                quest_id     = fbq.id,
                completed_on = week_start
            ))
            current_user.add_xp(fbq.xp_reward)
            flash(f"🏆 Quest complete: Feedback Champion! +{fbq.xp_reward} XP", 'success')
            db.session.commit()


    return redirect(url_for('bp.history'))

@bp.route('/trends')
@login_required
def trends():
    entries = Entry.query.filter_by(user_id=current_user.id).order_by(Entry.timestamp).all()

    daily_counts  = {}
    combined_dist = {}
    for e in entries:
        day = e.timestamp.astimezone().date().isoformat()
        daily_counts[day] = daily_counts.get(day, 0) + 1
        ce = e.combined_emotion or 'Unknown'
        combined_dist[ce] = combined_dist.get(ce, 0) + 1

    all_badges = Badge.query.order_by(Badge.threshold).all()
    # ── Monthly Word-Cloud ───────────────────────────────────────────────
    # collect all text_inputs from entries in the current month
    now = datetime.now(ZoneInfo('UTC')).astimezone()
    cm, cy = now.month, now.year
    texts = [
        e.text_input
        for e in entries
        if e.timestamp.astimezone().year == cy
        and e.timestamp.astimezone().month == cm
    ]
    wc_img = None
    if texts:
        combined = " ".join(texts)
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=STOPWORDS
        ).generate(combined)
        out_path = os.path.join(current_app.static_folder, "wordcloud.png")
        wc.to_file(out_path)
        wc_img = "wordcloud.png"
    
    # ── ANOMALY DETECTION ────────────────────────────────────────────────────
    # We'll look at the **current ISO-week** and the prior 8 full weeks:
    today     = datetime.now(tz=ZoneInfo('UTC')).date()
    # start of THIS week (Monday)
    this_monday = today - timedelta(days=today.weekday())

    # Count entries per week for the last N weeks, including current:
    N = 8
    week_counts = []
    for w in range(N):
        week_start = this_monday - timedelta(weeks=w)
        week_end   = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        cnt = (
            Entry.query
                 .filter(
                     Entry.user_id == current_user.id,
                     Entry.timestamp >= datetime.combine(week_start, datetime.min.time(), tzinfo=ZoneInfo('UTC')),
                     Entry.timestamp <= datetime.combine(week_end,   datetime.max.time(), tzinfo=ZoneInfo('UTC'))
                 )
                 .count()
        )
        week_counts.append(cnt)
    # week_counts[0] is **current** week; [1:] is history
    current_week_count = week_counts[0]
    history_counts     = week_counts[1:]

    anomaly = None
    # if we have at least 2 historical weeks to compute stdev
    if len(history_counts) >= 2:
        μ = statistics.mean(history_counts)
        σ = statistics.stdev(history_counts)
        # only if σ > 0
        if σ > 0:
            z = (current_week_count - μ) / σ
            # threshold at ±1.5σ
            if z < -1.5:
                anomaly = {
                    "level": "low",
                    "message": (
                        f"Your journaling this week ({current_week_count} entries) "
                        f"is well below your 8-week average ({μ:.1f}). "
                        "How about a quick entry now?"
                    )
                }
            elif z > 1.5:
                anomaly = {
                    "level": "high",
                    "message": (
                        f"You’ve written {current_week_count} entries this week—"
                        "that’s above your usual pace! Keep it going 🎉"
                    )
                }
    # ────────────────────────────────────────────────────────────────────────
    # ── 4) ENTRY-TOPIC CLUSTERING (“Your Themes”) ─────────────────────────
    # filter to this month’s texts for clustering
    theme_texts = texts[:]  # reused from above

    themes = []
    if len(theme_texts) >= 2:
        # 1) get embeddings
        embeddings = embedder.encode(theme_texts, show_progress_bar=False)

        # 2) choose k = sqrt(N) (at least 2)
        k = min(len(embeddings), max(2, int(len(embeddings)**0.5)))
        km = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = km.labels_

        # 3) group texts by cluster
        clusters = {}
        for lbl, txt in zip(labels, theme_texts):
            clusters.setdefault(lbl, []).append(txt)

        # 4) set up TF–IDF to extract top 2 phrases per cluster
        

        vect = TfidfVectorizer(
            ngram_range=(1,2),
            stop_words='english',
            max_features=100
        )

        for lbl, docs in clusters.items():
            # fit on just this cluster’s docs
            tfidf_matrix = vect.fit_transform(docs)
            # compute mean TF–IDF score per feature
            scores = tfidf_matrix.mean(axis=0).A1
            features = vect.get_feature_names_out()

            # pick top 2 scoring n-grams
            top_idxs = scores.argsort()[::-1][:2]
            top_phrases = [features[i].title() for i in top_idxs]

            themes.append({
                "label":    ", ".join(top_phrases),
                "examples": docs[:3]   # up to three sample entries
            })


    return render_template(
        'trends.html',
        daily_counts  = json.dumps(daily_counts),
        combined_dist = json.dumps(combined_dist),
        all_badges    = all_badges,
        wordcloud_img  = wc_img,
        wordcloud_month= now.strftime("%B %Y"),
        anomaly        = anomaly,
        themes        = themes
    )

@bp.route('/friends')
@login_required
def friends():
    # include current user first, then others
    users = User.query.order_by(User.id!=current_user.id).all()

    labels, entries_counts, helpful_counts = [], [], []
    one_week_ago = datetime.utcnow() - timedelta(days=7)

    for u in users:
        total = Entry.query.filter(
            Entry.user_id==u.id,
            Entry.timestamp>=one_week_ago
        ).count()
        helped = Entry.query.filter(
            Entry.user_id==u.id,
            Entry.timestamp>=one_week_ago,
            Entry.feedback.is_(True)
        ).count()

        labels.append("You" if u.id==current_user.id else u.email)
        entries_counts.append(total)
        helpful_counts.append(helped)

    return render_template(
        'friends.html',
        chart_labels    = json.dumps(labels),
        chart_entries   = json.dumps(entries_counts),
        chart_helpful   = json.dumps(helpful_counts),
    )
@bp.route('/quests')
@login_required
def quests():
    """Show the user’s daily & weekly quests and whether they’ve been completed."""
    today = date.today()
    # week start for weekly quest
    week_start = today - timedelta(days=today.weekday())

    # load all quests
    all_q = Quest.query.order_by(Quest.frequency).all()

    # load this user’s completions
    comps = QuestCompletion.query.filter_by(user_id=current_user.id).all()
    done = {(c.quest_id, c.completed_on) for c in comps}

    qs = []
    for q in all_q:
        if q.frequency == QuestFrequency.DAILY:
            key = (q.id, today)
        else:
            key = (q.id, week_start)
        qs.append({
            'name': q.name,
            'desc': q.description,
            'xp':   q.xp_reward,
            'freq': q.frequency.value,
            'done': key in done
        })

    return render_template('quests.html', quests=qs)

from transformers import pipeline
# …at the top of the file, after your other imports…
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")



@bp.route('/summary')
@login_required
def summary():
    # 1) PULL LAST 7 DAYS OF ENTRIES (This part remains unchanged)
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    recent = (
        Entry.query
             .filter(
                 Entry.user_id == current_user.id,
                 Entry.timestamp >= one_week_ago
             )
             .order_by(Entry.timestamp)
             .all()
    )

    # 2) IF NO ENTRIES, SHORT-CIRCUIT (This part remains unchanged)
    if not recent:
        return jsonify({
            'highlight':      "No entries yet—write a few this week to see your recap!",
            'recommendation': "Start today: jot down how you feel right now."
        })

    # 3) COUNT MOODS (This part remains unchanged)
    moods = [e.combined_emotion for e in recent if e.combined_emotion]
    counts = Counter(moods)
    top_mood, top_count = counts.most_common(1)[0]

    # 4) BUILD THE HIGHLIGHT SENTENCE (This part remains unchanged)
    highlight = (
        f"Over the past week you recorded “{top_mood}” "
        f"{top_count} time{'s' if top_count!=1 else ''}."
    )

    # --- 5) NEW: GENERATE SUGGESTION WITH OPENAI ---
    recommendation = ""
    try:
        # Set the API key from environment variables
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")

        # Create a very specific and concise prompt
        prompt = (
            f"A user's journal summary says: '{highlight}'. "
            f"Based only on this, write one short, positive, and forward-looking sentence of advice. "
            f"Be encouraging and gentle."
        )

        # Make the API call to the cheapest, fastest model
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly and supportive journaling assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Strictly limit the output length to save credits
            temperature=0.7,  # A good balance of creative but not too random
            n=1,
            stop=None
        )
        recommendation = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        # --- 6) FALLBACK: Use the original dictionary if the API fails ---
        # This makes your application robust and saves you from errors.
        suggestions = {
            'joy':     "Keep savoring those good moments—maybe share one with a friend?",
            'sadness': "It might help to take a 5-minute break or talk to someone you trust.",
            'fear':    "Try a quick breathing exercise when you feel anxious.",
            'disgust': "A short walk or change of scenery can help reset your perspective.",
            'neutral': "Consistency is key—keep journaling to spot your patterns.",
        }
        recommendation = suggestions.get(
            top_mood.lower(),
            "Keep up the habit—your insights will grow over time."
        )

    # --- 7) RETURN THE FINAL JSON (This part remains unchanged) ---
    return jsonify({
      'highlight':      highlight,
      'recommendation': recommendation
    })
