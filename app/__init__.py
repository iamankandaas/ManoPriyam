# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
from zoneinfo import ZoneInfo

# Extensions
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'login'

def seed_quests():
    """
    Create the two base quests (daily & weekly) if they don't exist yet.
    Called once right after tables are created.
    """
    from app.models import Quest, QuestFrequency

    # If any quests exist, assume seeding already done
    if Quest.query.first():
        return

    specs = [
        ("Daily Check‑In",      QuestFrequency.DAILY,  20,  "Log at least one entry today."),
        ("Weekly Streak‑Keeper", QuestFrequency.WEEKLY, 100, "Maintain your streak for 7 days."),
        ("Feedback Champion", QuestFrequency.WEEKLY, 50, "Give feedback on at least 5 entries this week."),
        ("Consistent Trio",        QuestFrequency.WEEKLY,  50,  "Maintain your streak for 3 days."),
        ("Level 2 Achiever",       QuestFrequency.WEEKLY,  30,  "Reach Level 2.")
    ]

    for name, freq, xp_reward, desc in specs:
        q = Quest(
            name=name,
            frequency=freq,
            xp_reward=xp_reward,
            description=desc
        )
        db.session.add(q)

    db.session.commit()


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Register your routes blueprint
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Jinja filter to convert UTC datetimes → IST strings
    @app.template_filter('localtime')
    def format_local(dt, fmt='%Y-%m-%d %H:%M'):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        return dt.astimezone(ZoneInfo('Asia/Kolkata')).strftime(fmt)

    # Create all tables, then seed quests
    with app.app_context():
        db.create_all()
        seed_quests()

    return app


# Flask‑Login user loader
from app.models import User
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
