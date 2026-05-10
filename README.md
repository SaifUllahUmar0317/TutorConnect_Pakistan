# TutorConnect Pakistan 🇵🇰

🎓 **Bridging university tutors with families – AI‑powered, donor‑funded, and free for students.**

TutorConnect Pakistan is a full‑stack web platform that connects university students (tutors) with families needing academic support. It features a **hybrid AI chatbot** combining **RAG (Retrieval‑Augmented Generation)** and **Groq LLM** for intelligent, context‑aware conversations, along with registration dashboards for both tutors and families.

---

## ✨ Key Features

- **AI Chatbot**  
  - Routes queries automatically: general chat → Groq LLM, platform‑specific → RAG retrieval + paraphrasing.  
  - Remembers conversation history per session (up to 10 exchanges).  
  - Escalation to human support.

- **User Registration**  
  - Tutors: submit personal info, CNIC, university transcript, subjects, areas, availability.  
  - Families: provide child’s details, required subjects, preferred timings.

- **Dashboards**  
  - Tutor dashboard: profile status, earnings, upcoming sessions.  
  - Family dashboard: child info, AI‑matched tutors, session history.

- **Persistent Storage**  
  - Frontend: `localStorage` for instant dashboard display.  
  - Backend: JSON files (`family_registrations.json`, `tutor_registrations.json`) for long‑term analysis.  
  - SQLite database (`tutorconnect_logs.db`) logs every user‑bot interaction.

- **RAG Pipeline**  
  - Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.  
  - ChromaDB as the vector store (knowledge base from `knowledge_base.json`).

- **Static File Serving**  
  - Flask serves all HTML, CSS, and JS directly – no separate web server needed.

---

## 🛠️ Tech Stack

| Category         | Technologies                                                                 |
|------------------|------------------------------------------------------------------------------|
| **Backend**      | Python 3.9+, Flask, Flask‑CORS                                               |
| **AI / NLP**     | Groq LLM (`llama-3.3-70b-versatile`), sentence‑transformers, ChromaDB          |
| **Frontend**     | HTML5, Tailwind CSS, Vanilla JavaScript                                      |
| **Storage**      | SQLite (logs), JSON files (registrations), localStorage (client)            |
| **Environment**  | `python-dotenv` for API keys                                                 |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/tutorconnect-pakistan.git
cd tutorconnect-pakistan
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyTorch will be installed automatically. If you encounter issues, use the CPU‑only version:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Set environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key from [GroqCloud](https://console.groq.com).

### 5. Run the application
```bash
python app.py
```
The server will start at `http://localhost:5000`.

---

## 🧪 Usage

1. **Open your browser** at `http://localhost:5000`.  
2. **Register** as a tutor or a family using the navigation bar.  
3. **Chat** with the AI assistant (bottom‑right button) – ask about registration, subjects, stipends, or just say “hi”.  
4. **View dashboards** after registration.  
5. **Data is saved** in:  
   - `tutor_registrations.json` / `family_registrations.json`  
   - `tutorconnect_logs.db` (interactions and escalations)

---

## 📁 Project Structure

```
TutorConnectPakistan/
├── app.py                     # Flask backend (RAG + Groq + memory)
├── knowledge_base.json        # Domain‑specific Q&A for RAG
├── requirements.txt           # Python dependencies
├── .env                       # Groq API key (ignored by git)
├── chroma_db/                 # ChromaDB vector store (auto‑generated)
├── tutorconnect_logs.db       # SQLite interaction logs
├── family_registrations.json  # Saved family registrations
├── tutor_registrations.json   # Saved tutor registrations
├── index.html                 # Landing page
├── family-register.html       # Family registration form
├── tutor-register.html        # Tutor registration form
├── family-dashboard.html      # Family dashboard
├── tutor-dashboard.html       # Tutor dashboard
├── css/
│   └── style.css              # Custom styles
└── js/
    ├── chatbot.js             # Chat widget frontend
    └── main.js                # Form handling, localStorage, and API calls
```

---

## 🤝 How to Contribute

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`feature/your-feature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for providing fast LLM inference.
- [Hugging Face](https://huggingface.co) for `sentence-transformers` models.
- [Chroma](https://www.trychroma.com) for the vector database.
- All donors and NGOs who make free tutoring possible.

---

## 📬 Contact

For questions or support, please open an issue on GitHub or email: **support@tutorconnect.pk**

---

**Made with ❤️ for Pakistan’s students and educators.**
