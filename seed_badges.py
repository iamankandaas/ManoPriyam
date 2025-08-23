# seed_badges.py
from app import create_app, db
from app.models import Badge

app = create_app()
with app.app_context():
    for days, name in [(1, 'First Entry'), (7,'First Week'), (30,'Month Master'), (90,'90‑Day Hero')]:
        if not Badge.query.filter_by(threshold=days).first():
            db.session.add(
              Badge(
                name=name,
                threshold=days,
                description=f'A {days}-day logging streak!'
              )
            )
    db.session.commit()
    print("Badges seeded:", Badge.query.all())
