# Manopriyam 🪷

*AI-Powered Mental Health Journaling Assistant*

> “This project is more than just code — it’s a reflection of lived experiences, curiosity, and the pursuit of building something meaningful.”

---

## 🌟 Overview

**Manopriyam** is an AI-driven journaling platform that blends **emotional intelligence**, **personalization**, and **gamification** to help users reflect on their mental health.

Built from the ground up with a focus on **students and young adults**, Manopriyam goes beyond generic chatbots. It combines **text-based emotion analysis** with **facial expression recognition**, offers empathetic AI-generated suggestions, and motivates consistency through a thoughtfully designed reward system.

This project is a culmination of months of **data curation, fine-tuning, and careful engineering choices** — a deeply personal attempt to merge technical expertise with a real-world problem close to my heart.

---

## ✨ Key Features

* **📝 Daily Journaling (Multi-Modal)**

  * Text entries + optional webcam capture for richer context.
* **🤖 Emotion Analysis (Ensemble AI)**

  * Fine-tuned transformer models with **80%+ F1 score**.
  * Weighted fusion of text (70%) and face (30%) emotion signals.
* **💬 Personalized AI Suggestions**

  * Context-aware, empathetic replies powered by OpenAI.
  * Tailored advice woven with user’s hobbies and interests.
* **🎮 Gamification Layer**

  * XP, levels, streaks, badges, and daily/weekly quests.
* **📊 Self-Reflection & Insights**

  * Monthly word clouds.
  * Journaling anomaly alerts.
  * Automatic theme clustering via embeddings + KMeans.
* **📜 History & Filters**

  * Clean review of past entries with advanced filters.

---

## 🧠 The AI Journey

This project wasn’t just about coding — it was about **researching, iterating, and learning**.

* **Baseline Models (DistilBERT, BERTweet)** → 74.6% F1.
* **Domain-Specific Training (Twitter-RoBERTa)** → Stronger performance on real-world student data.
* **Data-Centric Innovations**: emoji-to-text, class weights, back-translation, label cleaning.
* **Augmentation Experiments**: lessons learned that quality > quantity.
* **Ensemble Breakthrough**: combined two champion models → **81.5% accuracy, 0.8142 F1**, a milestone in nuanced emotion detection.

This iterative journey mirrors **how real-world AI products are built** — by refining data, testing assumptions, and making deliberate trade-offs.

---

## 🏗️ Technical Stack

* **Backend**: Flask + SQLAlchemy + Gunicorn
* **Frontend**: HTML, CSS, Vanilla JS (Webcam integration)
* **AI/ML**:

  * Transformers (Hugging Face) + PyTorch
  * TensorFlow (FER)
  * Sentence-Transformers + Scikit-learn (clustering)
  * OpenAI API (context-aware suggestions)
* **Database**: SQLite
* **Deployment**: Dockerized app on Render (PaaS)

---

## 🚧 Challenges & Learnings

* **Data Scarcity** → Built my own dataset, cleaned, balanced, augmented.
* **Dependency Conflicts** → Solved by Dockerizing.
* **Resource Constraints** → Optimized models to run within budget limits.
* **Deployment Limits** → Free-tier servers couldn’t handle full transformer stacks → adapted by providing recorded demo.

Each of these hurdles added **practical industry-level experience** in MLOps, deployment, and data-centric AI.

---

## 🎥 Demo

A complete demo of Manopriyam is available here:
👉 [Demo Video Link](#) *(Google Drive / YouTube)*

---

## 🔮 What’s Next

Time constraints kept me from scaling this to its full vision. Future possibilities include:

* **Cross-device accessibility** (mobile + web).
* **Blockchain-backed journaling** for **privacy & secure ownership** of entries.
* **Scalable deployment** on GPUs / cloud-native infrastructure.
* **More nuanced AI companions** for deeper emotional support.

---

## 💡 Personal Note

This project is not just a technical artifact. It’s something I poured my time, thought, and personal experiences into — **my “baby” project**.

While I had to cut down on some ambitious features due to time (most of which I spent curating high-quality data for fine-tuning), the outcome is something I’m **deeply proud of**. It demonstrates not only my technical depth but also my ability to carry an idea from a personal problem → to data → to model → to product.

---

## 📌 Takeaway

**Manopriyam** stands as proof that with the right blend of empathy, technical skill, and persistence, AI can be made **deeply personal, credible, and impactful**.

---

