"""
Run with:

    pytest -q

Offline tests only.
No API key or network required.

Tickets use an in-memory SQLite database.
"""

from datetime import datetime

import pytest

from rag_agent import (
    HelpDeskAgent,
    Retriever,
    TicketStore,
    chunk_text,
    parse_records
)


# ------------------------------------------------------------
# FIXED TEST TIMES
# ------------------------------------------------------------

TUE_10AM = datetime(
    2026,
    9,
    29,
    10,
    0
)


TUE_9PM = datetime(
    2026,
    9,
    29,
    21,
    0
)


SUN_10AM = datetime(
    2026,
    10,
    4,
    10,
    0
)


# ------------------------------------------------------------
# TEST AGENT
# ------------------------------------------------------------

@pytest.fixture()
def agent():

    return HelpDeskAgent(
        "college_info.txt",
        api_key=None,
        db_path=":memory:"
    )


# ------------------------------------------------------------
# RECORD TEST
# ------------------------------------------------------------

def test_parse_records_and_departments(agent):

    assert len(agent.departments) == 10

    assert "IT Support" in [
        department.name
        for department in agent.departments
    ]


# ------------------------------------------------------------
# CHUNKING TEST
# ------------------------------------------------------------

def test_chunking_skips_comments():

    assert (
        chunk_text(
            "# NOTE: skip\n\nHello."
        )
        ==
        ["Hello."]
    )


# ------------------------------------------------------------
# RETRIEVER TEST
# ------------------------------------------------------------

def test_retriever_relevance():

    retriever = Retriever(
        [
            "Bananas are yellow.",
            "Library fine is Rs 2 per day."
        ]
    )


    result = retriever.search(
        "library fine",
        k=1
    )


    assert (
        "Library"
        in
        result[0][0]
    )


# ------------------------------------------------------------
# GROQ OFFLINE TEST
# ------------------------------------------------------------

def test_offline_without_key(
    monkeypatch
):

    monkeypatch.delenv(
        "GROQ_API_KEY",
        raising=False
    )


    test_agent = HelpDeskAgent(
        "college_info.txt",
        api_key=None,
        db_path=":memory:"
    )


    assert test_agent.online is False


# ------------------------------------------------------------
# Q&A TEST
# ------------------------------------------------------------

def test_ask_answers_from_context(
    agent
):

    result = agent.ask(
        "What is the library fine per day?",
        now=TUE_10AM
    )


    assert result["sources"]


    assert (
        "Rs 2"
        in
        result["answer"]
    )


    assert (
        result["needs_ticket"]
        is False
    )


# ------------------------------------------------------------
# UNKNOWN QUESTION
# ------------------------------------------------------------

def test_ask_unknown_suggests_ticket(
    agent
):

    result = agent.ask(
        "zzzz qqqq xxxx",
        now=TUE_10AM
    )


    assert (
        result["needs_ticket"]
        is True
    )


# ------------------------------------------------------------
# DEPARTMENT ROUTING
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "text,dept",
    [

        (
            "I cannot login to the student portal",
            "IT Support"
        ),

        (
            "No hot water in my hostel room",
            "Hostel Office"
        ),

        (
            "I paid my fees but did not get a receipt",
            "Accounts Office"
        ),

        (
            "The college bus was late again",
            "Transport Office"
        ),

        (
            "I need a bonafide certificate",
            "Administration Office"
        ),

    ]
)
def test_routing(
    agent,
    text,
    dept
):

    assert (
        agent.route_department(text).name
        ==
        dept
    )


# ------------------------------------------------------------
# DEFAULT ROUTING
# ------------------------------------------------------------

def test_routing_falls_back_to_default(
    agent
):

    department = agent.route_department(
        "qwerty asdf zxcv"
    )


    assert (
        department.name
        in
        [
            d.name
            for d in agent.departments
        ]
    )


# ------------------------------------------------------------
# PRIORITY TEST
# ------------------------------------------------------------

def test_priority_levels(
    agent
):

    assert (
        agent.detect_priority(
            "There is a fire near the lab"
        )
        ==
        "High"
    )


    assert (
        agent.detect_priority(
            "The projector is not working"
        )
        ==
        "Medium"
    )


    assert (
        agent.detect_priority(
            "Please share the bus timing"
        )
        ==
        "Low"
    )


# ------------------------------------------------------------
# CREATE TICKET
# ------------------------------------------------------------

def test_create_and_get_ticket(
    agent
):

    ticket = agent.create_ticket(
        "Asha",
        "asha@x.edu",
        "Wifi is not working in library"
    )


    assert (
        ticket["ticket_id"]
        ==
        "TKT-0001"
    )


    assert (
        ticket["status"]
        ==
        "Open"
    )


    assert (
        ticket["department"]
        ==
        "IT Support"
    )


    assert (
        ticket["department_email"]
        .endswith("edu")
    )


    assert (
        agent.tickets.get(
            "tkt-0001"
        )["name"]
        ==
        "Asha"
    )


# ------------------------------------------------------------
# EMPTY DESCRIPTION
# ------------------------------------------------------------

def test_empty_description_rejected(
    agent
):

    with pytest.raises(ValueError):

        agent.create_ticket(
            "A",
            "a@x.edu",
            "   "
        )


# ------------------------------------------------------------
# STATUS UPDATE
# ------------------------------------------------------------

def test_status_update_and_validation(
    agent
):

    ticket = agent.create_ticket(
        "A",
        "a@x.edu",
        "Broken fan in hostel room"
    )


    updated = (
        agent.tickets.update_status(
            ticket["ticket_id"],
            "Resolved",
            "Fan replaced"
        )
    )


    assert (
        updated["status"]
        ==
        "Resolved"
    )


    assert (
        updated["note"]
        ==
        "Fan replaced"
    )


    with pytest.raises(ValueError):

        agent.tickets.update_status(
            ticket["ticket_id"],
            "Banana"
        )


    assert (
        agent.tickets.update_status(
            "TKT-9999",
            "Open"
        )
        is None
    )


# ------------------------------------------------------------
# LIST FILTERS
# ------------------------------------------------------------

def test_list_filters(
    agent
):

    agent.create_ticket(
        "A",
        "a@x.edu",
        "Wifi down"
    )


    agent.create_ticket(
        "B",
        "b@x.edu",
        "Bus late"
    )


    assert (
        len(agent.tickets.list())
        ==
        2
    )


    assert (
        len(
            agent.tickets.list(
                contact="A@X.EDU"
            )
        )
        ==
        1
    )


    assert (
        agent.tickets.list(
            status="Resolved"
        )
        ==
        []
    )


# ------------------------------------------------------------
# OPEN/CLOSED LOGIC
# ------------------------------------------------------------

def test_is_open_logic(
    agent
):

    library = next(
        d

        for d in agent.departments

        if d.name == "Library"
    )


    security = next(
        d

        for d in agent.departments

        if d.name == "Security Office"
    )


    assert (
        library.is_open(TUE_10AM)
        and
        not library.is_open(TUE_9PM)
        and
        not library.is_open(SUN_10AM)
    )


    assert (
        security.is_open(
            SUN_10AM
        )
    )


# ------------------------------------------------------------
# DIRECTORY TEST
# ------------------------------------------------------------

def test_directory_has_open_flag(
    agent
):

    rows = agent.directory(
        TUE_10AM
    )


    assert all(
        "Open now" in row
        for row in rows
    )


    assert (
        len(rows)
        ==
        10
    )