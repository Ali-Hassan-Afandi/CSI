
import json
import os

from datetime import datetime

import pandas as pd
import streamlit as st

from groq import Groq

from rag_engine import (
    ForensicRAG,
    make_context
)


MODEL = "openai/gpt-oss-20b"


st.set_page_config(

    page_title=
        "SceneGuard AI v2",

    page_icon=
        "🧪",

    layout=
        "wide"
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():

    try:

        if "GROQ_API_KEY" in st.secrets:

            return st.secrets[
                "GROQ_API_KEY"
            ]

    except Exception:

        pass


    return os.getenv(
        "GROQ_API_KEY",
        ""
    )


# ============================================================
# GROQ JSON
# ============================================================

def groq_json(
    system_prompt,
    user_prompt
):

    key = get_api_key()


    if not key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )


    client = Groq(
        api_key=key
    )


    response = (

        client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt
                }
            ],

            temperature=0.1,

            response_format={
                "type":
                    "json_object"
            },

            max_tokens=2200
        )
    )


    return json.loads(

        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# GROQ TEXT
# ============================================================

def groq_text(
    system_prompt,
    user_prompt
):

    key = get_api_key()


    if not key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )


    client = Groq(
        api_key=key
    )


    response = (

        client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt
                }
            ],

            temperature=0.15,

            max_tokens=2200
        )
    )


    return (

        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# LOAD RAG ONCE
# ============================================================

@st.cache_resource(
    show_spinner=
        "Building official forensic knowledge index..."
)
def load_rag():

    return ForensicRAG(
        "knowledge"
    )


# ============================================================
# AGENT 1
# SCENE ANALYSIS AGENT
# ============================================================

def scene_agent(
    scene_type,
    description
):

    system = """
You are SceneGuard's Scene Analysis Agent.

Your purpose is to structure reported crime-scene
observations for documentation support.

IMPORTANT:

Never determine guilt.

Never identify a suspect.

Never state that potential evidence has been
scientifically confirmed.

Use cautious terminology such as:

possible
potential
reported
apparent

Return valid JSON only.

Schema:

{
 "scene_summary": "...",

 "hazards": [
   "..."
 ],

 "potential_evidence": [
   {
     "item": "...",
     "category":
       "biological|digital|latent_print|trace|physical|other",
     "location": "...",
     "reason": "..."
   }
 ],

 "immediate_documentation_priorities": [
   "..."
 ]
}
"""


    prompt = f"""
SCENE TYPE:

{scene_type}

SCENE DESCRIPTION:

{description}
"""


    return groq_json(
        system,
        prompt
    )


# ============================================================
# AGENT 2
# SOP RETRIEVAL AGENT
# ============================================================

def retrieval_agent(

    rag,
    scene_type,
    description,
    analysis
):

    evidence_terms = ", ".join(

        f"{item.get('item','')} "
        f"({item.get('category','other')})"

        for item
        in analysis.get(
            "potential_evidence",
            []
        )
    )


    retrieval_query = f"""
Official National Forensics Agency Pakistan
evidence collection packaging preservation
documentation submission and chain-of-custody
guidance.

Scene type:
{scene_type}

Scene observations:
{description}

Potential evidence:
{evidence_terms}
"""


    # Vector database tool call
    results = rag.search(

        retrieval_query,

        k=7
    )


    context = make_context(
        results
    )


    system = """
You are SceneGuard's SOP Retrieval Agent.

You have retrieved excerpts from official
National Forensics Agency Pakistan sources.

RULES:

1. Answer ONLY from supplied excerpts.

2. Never invent an SOP.

3. Never convert possible evidence into
scientifically confirmed evidence.

4. Distinguish general guidance from
discipline-specific requirements.

5. Every material procedural statement must
contain a citation such as [S1], [S2], [S3].

6. If the indexed official documents do not
establish something, explicitly say:

"The indexed official NFA sources do not
establish this requirement."

Prepare:

A. Immediate Documentation Priorities

B. Evidence-Specific Guidance

C. Packaging / Preservation Considerations

D. Chain-of-Custody / Submission Considerations

E. Source Limitations
"""


    prompt = f"""
SCENE DESCRIPTION:

{description}


OFFICIAL NFA SOURCE EXCERPTS:

{context}
"""


    answer = groq_text(

        system,

        prompt
    )


    return (
        answer,
        results
    )


# ============================================================
# BUILD EVIDENCE INVENTORY
# ============================================================

def evidence_table(
    analysis
):

    rows = []


    for number, item in enumerate(

        analysis.get(
            "potential_evidence",
            []
        ),

        start=1
    ):

        rows.append({

            "Evidence ID":
                f"E-{number:03d}",

            "Item":
                item.get(
                    "item",
                    "Unknown item"
                ),

            "Category":
                item.get(
                    "category",
                    "other"
                ),

            "Location":
                item.get(
                    "location",
                    "Not stated"
                ),

            "Photographed":
                False,

            "Collector Recorded":
                False,

            "Packaging Recorded":
                False,

            "Seal Recorded":
                False,

            "Custody Started":
                False
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# AGENT 3
# EVIDENCE INTEGRITY AGENT
#
# Deterministic Python - NOT LLM.
# ============================================================

def integrity_scores(
    dataframe
):

    checks = [

        "Photographed",

        "Collector Recorded",

        "Packaging Recorded",

        "Seal Recorded",

        "Custody Started"
    ]


    if dataframe.empty:

        return (
            0,
            []
        )


    completed = int(

        dataframe[
            checks
        ]
        .sum()
        .sum()
    )


    possible = (

        len(dataframe)
        *
        len(checks)
    )


    score = round(

        completed
        /
        possible
        *
        100
    )


    alerts = []


    for _, row in dataframe.iterrows():

        missing = [

            check

            for check
            in checks

            if not bool(
                row[check]
            )
        ]


        if missing:

            alerts.append(

                f"{row['Evidence ID']} – "
                f"{row['Item']}: missing "
                f"{', '.join(missing)}"
            )


    return (
        score,
        alerts
    )


# ============================================================
# AGENT 4
# REPORT AGENT
# ============================================================

def report_agent(

    case_metadata,
    analysis,
    sop_guidance,
    sources,
    evidence_dataframe,
    score,
    notes
):

    official_context = (
        make_context(
            sources
        )
    )


    system = """
You are SceneGuard's Crime Scene Report Agent.

Prepare a professional crime-scene
documentation DRAFT.

IMPORTANT:

Never determine guilt.

Never identify a suspect.

Never claim scientific confirmation of
suspected material.

Procedural statements must be grounded in
the supplied official NFA sources.

Use [S1], [S2], etc. citations.

Clearly separate:

- user/investigator observations
- AI-extracted potential evidence
- official-source procedural guidance
- documentation gaps
- investigator notes

Include:

CASE DETAILS

SCENE SUMMARY

POTENTIAL EVIDENCE INVENTORY

SOURCE-GROUNDED PROCEDURAL GUIDANCE

DOCUMENTATION GAPS

INVESTIGATOR NOTES

SOURCE REFERENCES

DISCLAIMER
"""


    payload = {

        "case":
            case_metadata,

        "scene_analysis":
            analysis,

        "sop_guidance":
            sop_guidance,

        "evidence_inventory":
            evidence_dataframe.to_dict(
                orient="records"
            ),

        "integrity_score":
            score,

        "investigator_notes":
            notes
    }


    prompt = f"""
CASE DATA:

{json.dumps(
    payload,
    default=str,
    indent=2
)}

OFFICIAL SOURCE EXCERPTS:

{official_context}
"""


    return groq_text(

        system,

        prompt
    )


# ============================================================
# UI
# ============================================================

st.title(
    "🧪 SceneGuard AI v2"
)


st.subheader(
    "Agentic Crime Scene Investigation Assistant"
)


st.caption(
    "Official-source RAG prototype "
    "| National Forensics Agency Pakistan"
)


st.warning(
    "Hackathon and training prototype. "
    "Use fictional or anonymized case information only. "
    "Retrieved guidance assists documentation and does "
    "not replace official agency SOPs, forensic experts, "
    "laboratory protocols or legal requirements."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "SceneGuard v2"
    )


    st.write(
        "**Generative Model**"
    )

    st.code(
        MODEL
    )


    st.write(
        "**Embedding Model**"
    )

    st.code(
        "all-MiniLM-L6-v2"
    )


    st.write(
        "**Vector Database**"
    )

    st.code(
        "FAISS"
    )


    if get_api_key():

        st.success(
            "Groq API connected"
        )

    else:

        st.error(
            "GROQ_API_KEY missing"
        )


    st.divider()


    st.write(
        "### Agents"
    )

    st.write(
        "1️⃣ Scene Analysis Agent"
    )

    st.write(
        "2️⃣ SOP Retrieval Agent"
    )

    st.write(
        "3️⃣ Evidence Integrity Agent"
    )

    st.write(
        "4️⃣ Report Agent"
    )


# ============================================================
# STATE
# ============================================================

if "analysis" not in st.session_state:

    st.session_state.analysis = None


if "sop_guidance" not in st.session_state:

    st.session_state.sop_guidance = None


if "sources" not in st.session_state:

    st.session_state.sources = []


if "evidence_df" not in st.session_state:

    st.session_state.evidence_df = (
        pd.DataFrame()
    )


# ============================================================
# SCENE INTAKE
# ============================================================

st.header(
    "1️⃣ Scene Intake"
)


column1, column2, column3 = (
    st.columns(3)
)


with column1:

    case_id = st.text_input(

        "Case / Scene ID",

        "SG-DEMO-002"
    )


with column2:

    scene_type = st.selectbox(

        "Scene Type",

        [
            "Burglary",
            "Assault",
            "Vehicle",
            "Fire",
            "Digital Device Recovery",
            "Other"
        ]
    )


with column3:

    scene_time = st.text_input(

        "Date / Time",

        datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )
    )


location = st.text_input(

    "Location",

    "Training Apartment, Room 2"
)


description = st.text_area(

    "Describe the scene",

    height=180,

    value=(
        "A broken bedroom window is visible. "
        "A reddish-brown stain is reported on a "
        "glass fragment near the window. "
        "A laptop is lying on the floor. "
        "Possible fingerprint marks are visible "
        "on a drinking glass on the table."
    )
)


# ============================================================
# RUN AGENTS
# ============================================================

if st.button(

    "🔎 Run Agentic Scene Analysis",

    type="primary"
):

    if not description.strip():

        st.error(
            "Please enter a scene description."
        )


    elif not get_api_key():

        st.error(
            "GROQ_API_KEY is missing."
        )


    else:

        try:

            # ------------------------------------------------
            # LOAD KNOWLEDGE TOOL
            # ------------------------------------------------

            rag = load_rag()


            # ------------------------------------------------
            # AGENT 1
            # ------------------------------------------------

            with st.spinner(
                "Scene Analysis Agent is "
                "structuring observations..."
            ):

                analysis = scene_agent(

                    scene_type,

                    description
                )


            # ------------------------------------------------
            # AGENT 2
            # ------------------------------------------------

            with st.spinner(
                "SOP Retrieval Agent is searching "
                "official NFA sources..."
            ):

                sop_guidance, sources = (
                    retrieval_agent(

                        rag,

                        scene_type,

                        description,

                        analysis
                    )
                )


            st.session_state.analysis = (
                analysis
            )


            st.session_state.sop_guidance = (
                sop_guidance
            )


            st.session_state.sources = (
                sources
            )


            st.session_state.evidence_df = (
                evidence_table(
                    analysis
                )
            )


            st.success(
                "All scene agents completed."
            )


        except Exception as error:

            st.exception(
                error
            )


# ============================================================
# RESULTS
# ============================================================

analysis = (
    st.session_state.analysis
)


if analysis:

    st.divider()


    # ========================================================
    # SCENE AGENT OUTPUT
    # ========================================================

    st.header(
        "2️⃣ Scene Analysis Agent"
    )


    st.info(
        analysis.get(
            "scene_summary",
            ""
        )
    )


    left, right = st.columns(2)


    with left:

        st.subheader(
            "⚠️ Potential Hazards"
        )


        hazards = analysis.get(
            "hazards",
            []
        )


        if hazards:

            for hazard in hazards:

                st.write(
                    f"• {hazard}"
                )

        else:

            st.write(
                "No hazards extracted."
            )


    with right:

        st.subheader(
            "📷 Documentation Priorities"
        )


        for priority in analysis.get(

            "immediate_documentation_priorities",

            []
        ):

            st.write(
                f"• {priority}"
            )


    # ========================================================
    # RAG AGENT
    # ========================================================

    st.header(
        "3️⃣ SOP Retrieval Agent"
    )


    st.caption(
        "Recommendations below are generated "
        "from retrieved NFA source excerpts."
    )


    st.markdown(

        st.session_state
        .sop_guidance
        or ""
    )


    # ========================================================
    # SOURCE CITATIONS
    # ========================================================

    with st.expander(

        "📚 Show retrieved official sources",

        expanded=False
    ):

        for number, source in enumerate(

            st.session_state.sources,

            start=1
        ):

            page_text = ""


            if source.get("page"):

                page_text = (

                    f" — Page "
                    f"{source['page']}"
                )


            st.markdown(

                f"### [S{number}] "
                f"{source['title']}"
                f"{page_text}"
            )


            st.caption(

                f"Authority: "
                f"{source['authority']}"
            )


            st.caption(

                f"Vector similarity: "
                f"{source['score']:.3f}"
            )


            st.write(
                source["text"]
            )


            st.link_button(

                "Open official NFA source",

                source["url"]
            )


            st.divider()


    # ========================================================
    # INTEGRITY AGENT
    # ========================================================

    st.header(
        "4️⃣ Evidence Integrity Agent"
    )


    st.caption(
        "Tick a field only after that "
        "documentation action has actually occurred."
    )


    edited_dataframe = st.data_editor(

        st.session_state.evidence_df,

        use_container_width=True,

        hide_index=True,

        disabled=[

            "Evidence ID",

            "Item",

            "Category",

            "Location"
        ],

        key=
            "scene_guard_evidence_editor_v2"
    )


    st.session_state.evidence_df = (
        edited_dataframe
    )


    score, alerts = (
        integrity_scores(
            edited_dataframe
        )
    )


    score_column, alert_column = (
        st.columns(
            [1, 2]
        )
    )


    with score_column:

        st.metric(

            "Evidence Integrity Score",

            f"{score}%"
        )


        st.progress(
            score / 100
        )


    with alert_column:

        st.subheader(
            "Documentation Gaps"
        )


        if alerts:

            for alert in alerts:

                st.warning(
                    alert
                )


        else:

            st.success(
                "All tracked documentation "
                "fields are complete."
            )


    # ========================================================
    # REPORT AGENT
    # ========================================================

    st.header(
        "5️⃣ Source-Grounded Report Agent"
    )


    notes = st.text_area(

        "Investigator / Trainee Notes",

        placeholder=(
            "Add observations that should "
            "appear in the report..."
        )
    )


    if st.button(

        "📝 Generate Source-Grounded CSI Draft"
    ):

        metadata = {

            "case_id":
                case_id,

            "scene_type":
                scene_type,

            "date_time":
                scene_time,

            "location":
                location
        }


        try:

            with st.spinner(
                "Report Agent is generating "
                "the cited report..."
            ):

                report = report_agent(

                    metadata,

                    analysis,

                    st.session_state
                    .sop_guidance,

                    st.session_state
                    .sources,

                    edited_dataframe,

                    score,

                    notes
                )


            st.success(
                "Source-grounded report generated."
            )


            st.markdown(
                report
            )


            st.download_button(

                "⬇️ Download Report",

                data=report,

                file_name=(
                    f"{case_id}_"
                    f"SceneGuard_v2_Report.txt"
                ),

                mime=
                    "text/plain"
            )


        except Exception as error:

            st.exception(
                error
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "SceneGuard AI v2 | "
    "Human-in-the-loop | "
    "Official-source RAG | "
    "Hackathon prototype"
)
