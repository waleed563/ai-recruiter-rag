"""
AI Recruiter — Streamlit UI
===========================
Run with:
    streamlit run app.py

Make sure your FastAPI is running first:
    python -m uvicorn api:app --port 8000
"""

import streamlit as st
import requests
import time

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Recruiter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# Dark professional theme
# =========================

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111118;
        border-right: 1px solid #1e1e2e;
    }

    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: #111118 !important;
        border: 1px solid #2a2a3e !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7c6af7 !important;
        box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: #7c6af7 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: #6b58f0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(124, 106, 247, 0.3) !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111118 !important;
        border: 1px solid #1e1e2e !important;
        border-radius: 8px !important;
        color: #e8e8f0 !important;
    }

    /* Divider */
    hr {
        border-color: #1e1e2e !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# =========================
# API Config
# =========================

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def call_search(query, min_experience, top_n):
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query":          query,
                "min_experience": min_experience if min_experience > 0 else None,
                "top_n":          top_n
            },
            timeout=60
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure FastAPI is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def call_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except:
        return None


def call_candidates():
    try:
        r = requests.get(f"{API_URL}/candidates", timeout=10)
        return r.json()
    except:
        return None


# =========================
# Sidebar
# =========================

with st.sidebar:

    st.markdown("## 🎯 AI Recruiter")
    st.markdown("*Search 231 resumes with natural language*")
    st.divider()

    # Health check
    health = call_health()
    if health:
        st.success(f"✓ Connected — {health.get('total_chunks', 0):,} chunks")
    else:
        st.error("✗ API offline — start FastAPI first")

    st.divider()

    st.markdown("#### Search Settings")

    min_experience = st.slider(
        "Min. Years Experience",
        min_value=0,
        max_value=15,
        value=0,
        help="Filter candidates by minimum years of experience. Set to 0 for no filter."
    )

    top_n = st.slider(
        "Max Candidates",
        min_value=1,
        max_value=10,
        value=5,
        help="How many top candidates to return"
    )

    st.divider()

    # Quick example queries
    st.markdown("#### Quick Examples")

    examples = [
        "Java engineers with AWS and Docker",
        "Full stack developers with React",
        "Python developers with Django",
        "Senior architects with microservices",
        "BSA with Agile and Scrum",
    ]

    for example in examples:
        if st.button(example, key=f"ex_{example}"):
            st.session_state["query"] = example


# =========================
# Main Content
# =========================

# Header
st.markdown("""
<div style='padding: 2rem 0 1rem 0;'>
    <h1 style='font-family: DM Sans; font-size: 2.2rem; font-weight: 600;
               color: #e8e8f0; margin: 0;'>
        Find Your Next Hire
    </h1>
    <p style='color: #6b6b8a; font-size: 1rem; margin-top: 0.4rem;'>
        Describe the candidate you need in plain English
    </p>
</div>
""", unsafe_allow_html=True)

# Search input
query = st.text_area(
    label="Search Query",
    label_visibility="collapsed",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Find Java backend engineers with Spring Boot, AWS, and 5+ years of experience...",
    height=100,
    key="search_input"
)

search_clicked = st.button("🔍  Search Resumes")

st.divider()

# =========================
# Search Results
# =========================

if search_clicked and query.strip():

    with st.spinner("Searching resumes..."):
        start    = time.time()
        result   = call_search(query.strip(), min_experience, top_n)
        elapsed  = time.time() - start

    if "error" in result:
        st.error(f"Error: {result['error']}")

    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Candidates Found", result.get("total_found", 0))
        with col2:
            st.metric("Response Time", f"{result.get('time_taken', elapsed):.1f}s")
        with col3:
            exp_filter = f"{min_experience}+ years" if min_experience > 0 else "Any"
            st.metric("Experience Filter", exp_filter)

        st.divider()

        # AI Answer
        st.markdown("### AI Summary")
        st.markdown(
            f"<div style='background:#111118; border:1px solid #1e1e2e; "
            f"border-radius:10px; padding:1.25rem 1.5rem; "
            f"color:#c8c8d8; line-height:1.8; font-size:0.95rem;'>"
            f"{result.get('answer', '').replace(chr(10), '<br>')}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Candidate Cards
        candidates = result.get("candidates", [])
        if candidates:
            st.markdown("### Candidate Profiles")

            for i, candidate in enumerate(candidates, 1):
                name    = candidate.get("candidate_name", "Unknown")
                years   = candidate.get("years_experience", "N/A")
                email   = candidate.get("email", "N/A")
                skills  = candidate.get("skills", [])
                chunks  = candidate.get("relevant_chunks", 0)

                with st.expander(f"#{i}  {name}  —  {years} years experience", expanded=i == 1):

                    c1, c2 = st.columns([1, 2])

                    with c1:
                        st.markdown(f"**Name**")
                        st.markdown(f"{name}")
                        st.markdown(f"**Experience**")
                        st.markdown(f"{years} years")
                        st.markdown(f"**Email**")
                        st.markdown(f"{email}")
                        st.markdown(f"**Matched Chunks**")
                        st.markdown(f"{chunks} resume sections")

                    with c2:
                        st.markdown("**Skills**")
                        if skills:
                            # Show skills as tags
                            tags = " ".join([
                                f"<span style='background:#1e1e2e; color:#7c6af7; "
                                f"padding:3px 10px; border-radius:20px; "
                                f"font-size:12px; margin:2px; display:inline-block; "
                                f"font-family:DM Mono;'>{s}</span>"
                                for s in skills
                            ])
                            st.markdown(tags, unsafe_allow_html=True)
                        else:
                            st.markdown("No skills extracted")

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #3a3a5c;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🔍</div>
        <p style='font-size: 1.1rem;'>Enter a query above to search through 231 resumes</p>
        <p style='font-size: 0.9rem;'>Try the quick examples in the sidebar</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Candidates Browser tab
# =========================

with st.expander("📋  Browse All Candidates"):
    if st.button("Load All Candidates"):
        with st.spinner("Loading..."):
            data = call_candidates()
        if data:
            st.markdown(f"**{data['total']} candidates in database**")
            for c in data["candidates"]:
                name   = c.get("name", "Unknown")
                years  = c.get("years_experience", "?")
                skills = ", ".join(c.get("skills", [])[:5])
                st.markdown(
                    f"**{name}** — {years} yrs — `{skills}`"
                )
        else:
            st.error("Could not load candidates.")