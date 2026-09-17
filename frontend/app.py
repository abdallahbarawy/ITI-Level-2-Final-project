import streamlit as st

from api_client import APIConnectionError, APIError, ask_question, check_health, get_base_url

st.set_page_config(page_title="Football RAG Assistant", page_icon="ball", layout="centered")

st.title("Football Rules & Tactics Assistant")
st.caption(
    "Retrieval-augmented answers grounded in 12 curated football documents, "
    "served by a FastAPI backend with a local phi3:mini model."
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_error(message: str) -> None:
    st.error(
        f"**Sorry, something went wrong.**\n\n{message}\n\n"
        "Check that the backend is running (`uvicorn app.main:app --reload` in `backend/`) "
        f"and that `API_BASE_URL` in `frontend/.env` points to it. Current value: `{get_base_url()}`"
    )


with st.sidebar:
    st.subheader("Connection")
    try:
        health = check_health()
        st.success("Backend online")
        st.caption(f"`{get_base_url()}` — status `{health.get('status', 'ok')}`")
    except (APIConnectionError, APIError, RuntimeError) as exc:
        st.warning(f"Backend offline: {exc}")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

if question := st.chat_input("Ask about football rules and tactics..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents and thinking..."):
            try:
                result = ask_question(question)
            except APIConnectionError as exc:
                render_error(
                    f"{exc}. The assistant cannot reach the API at `{get_base_url()}`."
                )
            except APIError as exc:
                render_error(str(exc))
            else:
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- `{source}`")
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
