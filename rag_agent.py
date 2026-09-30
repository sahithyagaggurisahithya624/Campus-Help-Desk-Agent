"""
Campus Help Desk Agent - core logic.

Features:

1. Answers questions from college_info.txt
   using RAG + TF-IDF + Groq.

2. Routes help-desk tickets to departments.

3. Detects ticket priority.

4. Stores tickets in SQLite.

5. Tracks ticket status.

6. Shows department opening hours.
"""

from __future__ import annotations

import os
import re
import sqlite3

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

try:

    from dotenv import load_dotenv

    load_dotenv()

except ImportError:

    pass


# ------------------------------------------------------------
# GROQ CONFIGURATION
# ------------------------------------------------------------

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

STATUSES = (
    "Open",
    "In Progress",
    "Resolved"
)


FIELD_RE = re.compile(
    r"^([A-Za-z][A-Za-z ]*):\s*(.+)$"
)


DAY_NAMES = [
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun"
]


HIGH_WORDS = [
    "urgent",
    "emergency",
    "fire",
    "harass",
    "harassment",
    "ragging",
    "injury",
    "injured",
    "accident",
    "theft",
    "stolen",
    "unsafe",
    "threat",
    "ambulance",
    "fainted",
    "bleeding"
]


MEDIUM_WORDS = [
    "not working",
    "broken",
    "leak",
    "leaking",
    "no water",
    "power cut",
    "deadline",
    "exam",
    "failed",
    "cannot",
    "can't",
    "unable",
    "refund",
    "overcharged"
]


# ============================================================
# RECORD
# ============================================================

@dataclass
class Record:

    name: str

    raw: str

    fields: dict = field(
        default_factory=dict
    )


    @property
    def kind(self) -> str:

        return self.fields.get(
            "type",
            ""
        ).lower()


    @property
    def keywords(self) -> list[str]:

        return [
            keyword.strip().lower()

            for keyword in self.fields.get(
                "keywords",
                ""
            ).split(",")

            if keyword.strip()
        ]


    def is_open(
        self,
        now: datetime | None = None
    ) -> bool:

        now = now or datetime.now()


        # Check day

        if now.weekday() not in _parse_days(
            self.fields.get(
                "days",
                "Daily"
            )
        ):

            return False


        # Extract hours

        match = re.search(
            r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})",
            self.fields.get(
                "hours",
                ""
            )
        )


        # If no hours -> assume open

        if not match:

            return True


        h1, m1, h2, m2 = map(
            int,
            match.groups()
        )


        return (
            (h1, m1)
            <=
            (now.hour, now.minute)
            <=
            (h2, m2)
        )


# ============================================================
# DAY PARSER
# ============================================================

def _parse_days(
    spec: str
) -> set[int]:

    spec = spec.lower().strip()


    if spec in (
        "daily",
        "all",
        "everyday",
        "every day"
    ):

        return set(range(7))


    days = set()


    for part in spec.split(","):

        part = part.strip()


        if "-" in part:

            a, b = (
                p.strip()[:3]
                for p in part.split(
                    "-",
                    1
                )
            )


            if (
                a in DAY_NAMES
                and
                b in DAY_NAMES
            ):

                i = DAY_NAMES.index(a)
                j = DAY_NAMES.index(b)


                days |= {
                    (i + k) % 7

                    for k in range(
                        ((j - i) % 7) + 1
                    )
                }


        elif part[:3] in DAY_NAMES:

            days.add(
                DAY_NAMES.index(
                    part[:3]
                )
            )


    return days


# ============================================================
# PARSE COLLEGE RECORDS
# ============================================================

def parse_records(
    text: str
) -> list[Record]:

    records = []


    for block in re.split(
        r"\n\s*\n",
        text
    ):

        lines = block.strip().splitlines()


        if (
            not lines
            or
            not lines[0].startswith("## ")
        ):

            continue


        record = Record(
            name=lines[0][3:].strip(),
            raw=block.strip()
        )


        for line in lines[1:]:

            match = FIELD_RE.match(
                line.strip()
            )


            if match:

                record.fields[
                    match.group(1)
                    .strip()
                    .lower()
                ] = match.group(2).strip()


        if record.fields:

            records.append(record)


    return records


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    max_words: int = 120
) -> list[str]:

    chunks = []


    for block in re.split(
        r"\n\s*\n",
        text
    ):

        block = block.strip()


        if (
            not block
            or
            block.startswith("# ")
        ):

            continue


        words = block.split()


        if len(words) <= max_words:

            chunks.append(block)


        else:

            chunks += [
                " ".join(
                    words[i:i + max_words]
                )

                for i in range(
                    0,
                    len(words),
                    max_words
                )
            ]


    return chunks


# ============================================================
# TF-IDF RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        chunks: list[str]
    ):

        if not chunks:

            raise ValueError(
                "No text chunks to index."
            )


        self.chunks = chunks


        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )


        self.matrix = (
            self.vectorizer.fit_transform(
                chunks
            )
        )


    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.05
    ):

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )


        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).ravel()


        top = scores.argsort()[::-1][:k]


        return [
            (
                self.chunks[i],
                float(scores[i])
            )

            for i in top

            if scores[i] >= min_score
        ]


# ============================================================
# SQLITE TICKET STORE
# ============================================================

class TicketStore:

    def __init__(
        self,
        path: str = "tickets.db"
    ):

        self.conn = sqlite3.connect(
            path,
            check_same_thread=False
        )


        self.conn.row_factory = (
            sqlite3.Row
        )


        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL,

                contact TEXT NOT NULL,

                description TEXT NOT NULL,

                department TEXT NOT NULL,

                priority TEXT NOT NULL,

                status TEXT NOT NULL
                    DEFAULT 'Open',

                note TEXT DEFAULT '',

                created_at TEXT NOT NULL,

                updated_at TEXT NOT NULL
            )
            """
        )


        self.conn.commit()


    @staticmethod
    def _row(row):

        if row is None:

            return None


        data = dict(row)


        data["ticket_id"] = (
            f"TKT-{data['id']:04d}"
        )


        return data


    @staticmethod
    def _num(
        ticket_id
    ):

        digits = re.sub(
            r"\D",
            "",
            str(ticket_id)
        )


        return (
            int(digits)
            if digits
            else None
        )


    def create(
        self,
        name: str,
        contact: str,
        description: str,
        department: str,
        priority: str
    ) -> dict:

        if not description.strip():

            raise ValueError(
                "Description cannot be empty."
            )


        now = datetime.now().isoformat(
            timespec="seconds"
        )


        cursor = self.conn.execute(
            """
            INSERT INTO tickets
            (
                name,
                contact,
                description,
                department,
                priority,
                created_at,
                updated_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip() or "Anonymous",
                contact.strip(),
                description.strip(),
                department,
                priority,
                now,
                now
            )
        )


        self.conn.commit()


        return self.get(
            cursor.lastrowid
        )


    def get(
        self,
        ticket_id
    ) -> dict | None:

        number = self._num(
            ticket_id
        )


        if not number:

            return None


        row = self.conn.execute(
            "SELECT * FROM tickets WHERE id=?",
            (number,)
        ).fetchone()


        return self._row(row)


    def list(
        self,
        status: str | None = None,
        contact: str | None = None
    ) -> list[dict]:

        query = (
            "SELECT * FROM tickets WHERE 1=1"
        )


        args = []


        if status:

            query += " AND status=?"

            args.append(status)


        if contact:

            query += " AND lower(contact)=?"

            args.append(
                contact.strip().lower()
            )


        query += " ORDER BY id DESC"


        rows = self.conn.execute(
            query,
            args
        ).fetchall()


        return [
            self._row(row)
            for row in rows
        ]


    def update_status(
        self,
        ticket_id,
        status: str,
        note: str = ""
    ) -> dict | None:

        if status not in STATUSES:

            raise ValueError(
                f"Status must be one of {STATUSES}."
            )


        number = self._num(
            ticket_id
        )


        if (
            not number
            or
            not self.get(number)
        ):

            return None


        self.conn.execute(
            """
            UPDATE tickets

            SET
                status=?,
                note=?,
                updated_at=?

            WHERE id=?
            """,
            (
                status,
                note,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                number
            )
        )


        self.conn.commit()


        return self.get(
            number
        )


# ============================================================
# HELP DESK AGENT
# ============================================================

class HelpDeskAgent:

    def __init__(
        self,
        info_path: str = "college_info.txt",
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        db_path: str = "tickets.db"
    ):

        # ----------------------------------------------------
        # LOAD COLLEGE INFORMATION
        # ----------------------------------------------------

        text = Path(
            info_path
        ).read_text(
            encoding="utf-8"
        )


        # ----------------------------------------------------
        # PARSE RECORDS
        # ----------------------------------------------------

        self.records = parse_records(
            text
        )


        self.departments = [
            record

            for record in self.records

            if record.kind == "department"
        ]


        if not self.departments:

            raise ValueError(
                "college_info.txt must contain "
                "at least one 'Type: Department' section."
            )


        # ----------------------------------------------------
        # MAIN RETRIEVER
        # ----------------------------------------------------

        self.retriever = Retriever(
            chunk_text(text)
        )


        # ----------------------------------------------------
        # DEPARTMENT RETRIEVER
        # ----------------------------------------------------

        self._dept_by_raw = {
            department.raw: department

            for department in self.departments
        }


        self.dept_retriever = Retriever(
            [
                department.raw

                for department in self.departments
            ]
        )


        # ----------------------------------------------------
        # DEFAULT DEPARTMENT
        # ----------------------------------------------------

        self.default_dept = next(
            (
                department

                for department in self.departments

                if department.name
                == "Administration Office"
            ),
            self.departments[0]
        )


        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        self.tickets = TicketStore(
            db_path
        )


        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------

        self.model = model

        self.client = None


        key = (
            api_key
            or
            os.getenv("GROQ_API_KEY")
        )


        if key:

            try:

                from groq import Groq


                self.client = Groq(
                    api_key=key
                )

            except ImportError:

                self.client = None


    # ========================================================
    # ONLINE STATUS
    # ========================================================

    @property
    def online(self) -> bool:

        return self.client is not None


    # ========================================================
    # RAG + GROQ QUESTION ANSWERING
    # ========================================================

    def ask(
        self,
        question: str,
        k: int = 3,
        now: datetime | None = None
    ) -> dict:

        now = (
            now
            or
            datetime.now()
        )


        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        hits = self.retriever.search(
            question,
            k=k
        )


        sources = [
            hit[0]

            for hit in hits
        ]


        # ----------------------------------------------------
        # UNKNOWN QUESTION
        # ----------------------------------------------------

        if not sources:

            return {
                "answer": (
                    "I couldn't find that information "
                    "in the campus knowledge base.\n\n"
                    "Please raise a help desk ticket "
                    "so the appropriate department "
                    "can assist you."
                ),

                "sources": [],

                "needs_ticket": True
            }


        # ----------------------------------------------------
        # OFFLINE MODE
        # ----------------------------------------------------

        if not self.online:

            return {
                "answer": (
                    "(Offline mode - top matching "
                    "campus information)\n\n"
                    +
                    "\n\n".join(sources)
                ),

                "sources": sources,

                "needs_ticket": False
            }


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            f"[{i + 1}] {source}"

            for i, source in enumerate(
                sources
            )
        )


        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        system = (
            "You are a friendly campus help desk "
            "assistant. "

            "Answer ONLY from the provided context. "

            "Do not invent information. "

            "Include location, hours and contact "
            "details when relevant. "

            f"The current date and time is "
            f"{now.strftime('%A, %Y-%m-%d %H:%M')}. "

            "If the student asks whether an office "
            "is open now, use the supplied hours and "
            "days. "

            "If the answer is not present in the "
            "context, clearly say that the information "
            "was not found and suggest raising a "
            "help desk ticket. "

            "Be concise and helpful."
        )


        # ----------------------------------------------------
        # GROQ API CALL
        # ----------------------------------------------------

        try:

            response = (
                self.client
                .chat
                .completions
                .create(

                    model=self.model,

                    messages=[
                        {
                            "role": "system",
                            "content": system
                        },

                        {
                            "role": "user",
                            "content": (
                                f"Context:\n"
                                f"{context}\n\n"
                                f"Question: "
                                f"{question}"
                            )
                        }
                    ],

                    max_tokens=800,

                    temperature=0.2
                )
            )


            answer = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )


        except Exception as error:

            return {
                "answer": (
                    "The AI service could not be "
                    "reached right now.\n\n"
                    f"Retrieved information:\n\n"
                    f"{sources[0]}"
                ),

                "sources": sources,

                "needs_ticket": False,

                "error": str(error)
            }


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {
            "answer": answer,

            "sources": sources,

            "needs_ticket": False
        }


    # ========================================================
    # DEPARTMENT ROUTING
    # ========================================================

    def route_department(
        self,
        text: str
    ) -> Record:

        text_lower = text.lower()


        best = None
        best_key = None


        for department in self.departments:

            positions = []


            for keyword in department.keywords:

                match = re.search(
                    rf"\b{re.escape(keyword)}\b",
                    text_lower
                )


                if match:

                    positions.append(
                        match.start()
                    )


            if positions:

                key = (
                    len(positions),
                    -min(positions)
                )


                if (
                    best_key is None
                    or
                    key > best_key
                ):

                    best = department

                    best_key = key


        if best:

            return best


        # TF-IDF fallback

        hits = self.dept_retriever.search(
            text,
            k=1
        )


        if hits:

            return self._dept_by_raw[
                hits[0][0]
            ]


        return self.default_dept


    # ========================================================
    # PRIORITY DETECTION
    # ========================================================

    @staticmethod
    def detect_priority(
        text: str
    ) -> str:

        text_lower = text.lower()


        if any(
            re.search(
                rf"\b{re.escape(word)}",
                text_lower
            )

            for word in HIGH_WORDS
        ):

            return "High"


        if any(
            word in text_lower

            for word in MEDIUM_WORDS
        ):

            return "Medium"


        return "Low"


    # ========================================================
    # CREATE TICKET
    # ========================================================

    def create_ticket(
        self,
        name: str,
        contact: str,
        description: str
    ) -> dict:

        department = self.route_department(
            description
        )


        priority = self.detect_priority(
            description
        )


        ticket = self.tickets.create(
            name,
            contact,
            description,
            department.name,
            priority
        )


        ticket[
            "department_email"
        ] = department.fields.get(
            "email",
            "-"
        )


        ticket[
            "department_phone"
        ] = department.fields.get(
            "phone",
            "-"
        )


        return ticket


    # ========================================================
    # DEPARTMENT DIRECTORY
    # ========================================================

    def directory(
        self,
        now: datetime | None = None
    ) -> list[dict]:

        now = (
            now
            or
            datetime.now()
        )


        return [

            {
                "Department": department.name,

                "Location": department.fields.get(
                    "location",
                    "-"
                ),

                "Hours": department.fields.get(
                    "hours",
                    "-"
                ),

                "Days": department.fields.get(
                    "days",
                    "-"
                ),

                "Phone": department.fields.get(
                    "phone",
                    "-"
                ),

                "Email": department.fields.get(
                    "email",
                    "-"
                ),

                "Open now": department.is_open(
                    now
                )
            }

            for department in self.departments
        ]