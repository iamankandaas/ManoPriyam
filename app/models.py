# app/models.py

from datetime import datetime, date
from zoneinfo import ZoneInfo
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from enum import Enum

# Association table: which user has which badge
user_badges = db.Table(
    'user_badges',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('badge_id', db.Integer, db.ForeignKey('badge.id'), primary_key=True),
    db.Column('awarded_on', db.Date, default=date.today)
)

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id               = db.Column(db.Integer, primary_key=True)
    email            = db.Column(db.String(120), unique=True, nullable=False)
    password_hash    = db.Column(db.String(128), nullable=False)
    preferences      = db.Column(db.Text)  # store hobbies/preferences
    registered_on    = db.Column(db.DateTime, default=datetime.utcnow)

    # ── XP & leveling ──────────────────────────────────────────────────────
    xp               = db.Column(db.Integer, default=0, nullable=False)
    level            = db.Column(db.Integer, default=1, nullable=False)

    # ── streak fields ──────────────────────────────────────────────────────
    current_streak   = db.Column(db.Integer, default=0, nullable=False)
    last_entry_date  = db.Column(db.Date)


    # relationship to badges
    badges           = db.relationship('Badge', secondary=user_badges, backref='users')

    # thresholds for each level
    LEVEL_THRESHOLDS = {
        1:    0,    # Level 1 at 0 XP
        2:  100,    # Level 2 at 100 XP
        3:  250,
        4:  500,
        5: 1000,
    }

    def set_password(self, password):
        """Hashes and stores the password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks the provided password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def add_xp(self, amount: int):
        """Award XP, bump level if crossing threshold.
        Returns new level if leveled up, else None."""
        self.xp += amount
        new_level = self.level
        for lvl, thresh in sorted(self.LEVEL_THRESHOLDS.items()):
            if self.xp >= thresh:
                new_level = lvl
        if new_level > self.level:
            self.level = new_level
            return new_level
        return None

    @property
    def xp_for_next_level(self) -> int:
        """Total XP required to reach the next level."""
        # next level threshold (or same if none defined)
        next_lvl = self.level + 1
        return self.LEVEL_THRESHOLDS.get(next_lvl, self.LEVEL_THRESHOLDS[self.level])

    @property
    def xp_progress_percent(self) -> float:
        """What percent of the next-level XP the user has earned."""
        target = self.xp_for_next_level
        if target == 0:
            return 0.0
        pct = (self.xp / target) * 100
        return min(pct, 100.0)

class Badge(db.Model):
    __tablename__ = 'badge'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.String(200))
    threshold   = db.Column(db.Integer, nullable=False)  # e.g. days of streak

    def __repr__(self):
        return f'<Badge {self.name}@{self.threshold}d>'

class Entry(db.Model):
    __tablename__ = 'entry'
    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp          = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(tz=ZoneInfo('UTC'))
    )
    text_input         = db.Column(db.Text, nullable=False)
    text_emotion       = db.Column(db.String(64))
    text_confidence    = db.Column(db.Float)
    face_emotion       = db.Column(db.String(64))
    face_confidence    = db.Column(db.Float)
    combined_emotion   = db.Column(db.String(64))
    combined_confidence= db.Column(db.Float)
    feedback           = db.Column(db.Boolean, nullable=True)
    # <-- FINAL ADDITION: THIS IS THE ONLY CHANGE -->
    suggestion         = db.Column(db.Text, nullable=True)

    # relationship back to user
    user = db.relationship('User', backref=db.backref('entries', lazy='dynamic'))


class QuestFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"

class Quest(db.Model):
    __tablename__ = 'quest'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.String(200))
    frequency   = db.Column(db.Enum(QuestFrequency), nullable=False)
    xp_reward   = db.Column(db.Integer, default=20, nullable=False)

class QuestCompletion(db.Model):
    __tablename__ = 'quest_completion'
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quest_id  = db.Column(db.Integer, db.ForeignKey('quest.id'), nullable=False)
    completed_on  = db.Column(db.Date,    nullable=False)   # renamed from `date`
    week_year     = db.Column(db.Integer, nullable=True)    # for weekly quests
    week_number   = db.Column(db.Integer, nullable=True)

    user  = db.relationship('User', backref='quest_completions')
    quest = db.relationship('Quest')

    def __repr__(self):
        return f'<QuestCompletion quest={self.quest.name} on={self.completed_on}>'