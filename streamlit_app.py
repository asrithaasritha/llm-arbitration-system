"""
Streamlit UI -- the visual front end for the LLM Arbitration System.
Calls the FastAPI server over HTTP (keeps UI and backend decoupled).
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/v1/arbitrate"

st.set_page_config(page_title="LLM Arbitration System", layout="wide")

st.title("⚖️ LLM Arbitration System")
st.caption("3 independent AI critics evaluate an output — an adjudicator resolves their disagreements into one verdict.")

with st.form("arbitration_form"):
    original_question = st.text_input(
        "Original question (optional but recommended)",
        placeholder="e.g. What is the Eiffel Tower and when/where was it built?",
    )
    output_to_evaluate = st.text_area(
        "AI output to evaluate",
        height=150,
        placeholder="Paste the AI-generated answer you want judged...",
    )
    submitted = st.form_submit_button("Run Arbitration", use_container_width=True)


def score_color(score: int, max_score: int) -> str:
    """Simple red/yellow/green based on how good the score is."""
    ratio = score / max_score
    if ratio >= 0.7:
        return "green"
    elif ratio >= 0.4:
        return "orange"
    return "red"


if submitted:
    if not output_to_evaluate.strip():
        st.error("Please paste an output to evaluate.")
    else:
        with st.spinner("Running 3 critics in parallel, then adjudicating..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "original_question": original_question or None,
                        "output_to_evaluate": output_to_evaluate,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Make sure `uvicorn app.api:app` is running in another terminal.")
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        # ---- Critic comparison panel ----
        st.subheader("Critic Panel")
        cols = st.columns(3)
        dimension_labels = {"accuracy": "🎯 Accuracy", "logic": "🧠 Logic", "completeness": "📋 Completeness"}

        for col, critique in zip(cols, result["critiques"]):
            with col:
                color = score_color(critique["score"], 5)
                st.markdown(f"**{dimension_labels.get(critique['dimension'], critique['dimension'])}**")
                st.markdown(f":{color}[Score: {critique['score']}/5]")
                st.caption(f"Confidence: {critique['confidence']}/5")
                if critique["issues"]:
                    for issue in critique["issues"]:
                        st.markdown(f"- **{issue['claim']}**")
                        st.caption(issue["problem"])
                else:
                    st.success("No issues flagged")

        # ---- Disagreements ----
        st.subheader("Disagreements Detected")
        if result["disagreements"]:
            for d in result["disagreements"]:
                st.warning(d["description"])
        else:
            st.info("Critics were broadly aligned on this output.")

        # ---- Final verdict ----
        st.subheader("Final Verdict")
        verdict = result["verdict"]
        vcol1, vcol2 = st.columns([1, 3])
        with vcol1:
            st.metric("Overall Score", f"{verdict['overall_score']}/10")
            st.caption(f"Adjudicator confidence: {verdict['confidence']}/5")
        with vcol2:
            st.write(verdict["summary"])

        if verdict["confirmed_issues"]:
            with st.expander(f"✅ Confirmed Issues ({len(verdict['confirmed_issues'])})"):
                for issue in verdict["confirmed_issues"]:
                    st.markdown(f"**[{issue['source']}]** {issue['issue']} — severity {issue['severity']}/5")

        if verdict["dismissed_flags"]:
            with st.expander(f"❌ Dismissed False Alarms ({len(verdict['dismissed_flags'])})"):
                for flag in verdict["dismissed_flags"]:
                    st.markdown(f"**[{flag['source']}]** {flag['flag']}")
                    st.caption(f"Why dismissed: {flag['reason_dismissed']}")
