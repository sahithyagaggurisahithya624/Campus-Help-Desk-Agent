"""
Streamlit UI for the Campus Help Desk Agent.

Run:
    streamlit run app.py
"""

import os
import re

import pandas as pd
import streamlit as st

from rag_agent import STATUSES, HelpDeskAgent


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Campus Help Desk",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Campus Help Desk Agent")


# ------------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------------

with st.sidebar:

    st.header("Settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Leave empty to use offline mode."
    )

    st.divider()

    admin_pin = st.text_input(
        "Staff PIN (optional)",
        type="password",
        help="Staff can view all tickets and update their status."
    )


# ------------------------------------------------------------
# CREATE AGENT
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_agent(key: str) -> HelpDeskAgent:

    return HelpDeskAgent(
        "college_info.txt",
        api_key=key or None,
        db_path="tickets.db"
    )


agent = get_agent(api_key)


# ------------------------------------------------------------
# ADMIN CHECK
# ------------------------------------------------------------

is_admin = (
    bool(admin_pin)
    and admin_pin == os.getenv("ADMIN_PIN", "admin123")
)


if agent.online:

    st.sidebar.success("AI mode: ON")

else:

    st.sidebar.warning("Offline mode")


if is_admin:

    st.sidebar.info("Staff mode enabled")


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------

tab_chat, tab_ticket, tab_track, tab_dir = st.tabs(
    [
        "💬 Ask",
        "🎫 Raise a ticket",
        "🔎 Track tickets",
        "📞 Departments"
    ]
)


# ============================================================
# TAB 1 — ASK
# ============================================================

with tab_chat:

    st.subheader("💬 Campus Help Desk Assistant")

    st.caption(
        "Ask questions about campus facilities, departments, "
        "fees, library, hostel, transport, IT support, etc."
    )

    st.session_state.setdefault("messages", [])


    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # Chat input

    question = st.chat_input(
        "e.g. What time does the library close?"
    )


    if question:

        # Store user question

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Display user question

        with st.chat_message("user"):

            st.markdown(question)


        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner("Searching campus information..."):

                result = agent.ask(question)


            st.markdown(result["answer"])


            # Ticket suggestion

            if result["needs_ticket"]:

                st.warning(
                    "I could not find a reliable answer. "
                    "Please open the **Raise a ticket** tab "
                    "and submit your issue."
                )


            # Sources

            if result["sources"]:

                with st.expander("📚 Sources"):

                    for source in result["sources"]:

                        st.caption(source)


        # Store assistant answer

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )


# ============================================================
# TAB 2 — RAISE TICKET
# ============================================================

with tab_ticket:

    st.subheader("🎫 Raise a Help Desk Ticket")


    with st.form(
        "ticket_form",
        clear_on_submit=True
    ):

        c1, c2 = st.columns(2)


        name = c1.text_input(
            "Your name"
        )


        contact = c2.text_input(
            "Email or roll number",
            help="Used to look up your tickets later."
        )


        description = st.text_area(
            "Describe your issue",
            height=140,
            placeholder=(
                "Example: There is no hot water "
                "in Hostel Block A, room 214."
            )
        )


        submitted = st.form_submit_button(
            "Submit ticket",
            type="primary"
        )


    # Process ticket

    if submitted:

        if (
            not description.strip()
            or not contact.strip()
        ):

            st.error(
                "Please provide your contact "
                "and describe the issue."
            )

        else:

            try:

                ticket = agent.create_ticket(
                    name,
                    contact,
                    description
                )


                st.success(
                    f"Ticket **{ticket['ticket_id']}** "
                    f"created successfully."
                )


                st.info(
                    f"Department: **{ticket['department']}**"
                )


                st.info(
                    f"Priority: **{ticket['priority']}**"
                )


                st.caption(
                    f"Department email: "
                    f"{ticket['department_email']}"
                )


                st.caption(
                    f"Department phone: "
                    f"{ticket['department_phone']}"
                )


                st.warning(
                    f"Save your ticket ID: "
                    f"**{ticket['ticket_id']}**"
                )


            except Exception as e:

                st.error(
                    f"Could not create ticket: {e}"
                )


# ============================================================
# TAB 3 — TRACK TICKETS
# ============================================================

with tab_track:

    if is_admin:

        # ----------------------------------------------------
        # STAFF VIEW
        # ----------------------------------------------------

        st.subheader("👨‍💼 Staff Ticket Management")


        with st.form("update_form"):

            u1, u2, u3 = st.columns(
                [1, 1, 2]
            )


            ticket_id = u1.text_input(
                "Ticket ID",
                placeholder="TKT-0001"
            )


            status = u2.selectbox(
                "New status",
                STATUSES
            )


            note = u3.text_input(
                "Note to student"
            )


            update_button = st.form_submit_button(
                "Update Ticket"
            )


            if update_button:

                updated = agent.tickets.update_status(
                    ticket_id,
                    status,
                    note
                )


                if updated:

                    st.success(
                        f"{updated['ticket_id']} "
                        f"is now {updated['status']}"
                    )

                else:

                    st.error(
                        "Ticket not found."
                    )


        # Show all tickets

        rows = agent.tickets.list()


    else:

        # ----------------------------------------------------
        # STUDENT VIEW
        # ----------------------------------------------------

        st.subheader("🔎 Check Your Tickets")


        query = st.text_input(
            "Ticket ID or Email / Roll Number",
            placeholder="TKT-0001"
        )


        rows = []


        if query.strip():

            # Ticket ID search

            if re.match(
                r"(?i)^\s*(tkt-?)?\d+\s*$",
                query
            ):

                ticket = agent.tickets.get(query)

                rows = [ticket] if ticket else []


            # Contact search

            else:

                rows = agent.tickets.list(
                    contact=query
                )


            if not rows:

                st.info(
                    "No tickets found."
                )


    # --------------------------------------------------------
    # DISPLAY TICKETS
    # --------------------------------------------------------

    if rows:

        df = pd.DataFrame(rows)


        columns = [
            "ticket_id",
            "department",
            "priority",
            "status",
            "description",
            "note",
            "created_at",
            "updated_at"
        ]


        df = df[columns]


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 4 — DEPARTMENTS
# ============================================================

with tab_dir:

    st.subheader("📞 Campus Departments")


    department_data = agent.directory()


    df = pd.DataFrame(
        department_data
    )


    if not df.empty:

        df["Open now"] = df[
            "Open now"
        ].map(
            {
                True: "🟢 Open",
                False: "🔴 Closed"
            }
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No department information available."
        )