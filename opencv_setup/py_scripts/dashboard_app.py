import pandas as pd
import streamlit as st
from pathlib import Path

from dashboard_logger import LOG_FILE  # uses the same detections.csv


# -------- Data loading --------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the detections CSV into a DataFrame.
    Returns an empty DataFrame if the file doesn't exist or is empty.
    """
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "time",
                "source",
                "frame",
                "person_id",
                "x", "y", "w", "h",
                "age",
                "age_conf",
                "gender",
                "gender_conf",
                "ethnicity",
                "eth_conf",
            ]
        )

    try:
        df = pd.read_csv(LOG_FILE)

        # Basic dtype cleanup
        if "frame" in df.columns:
            df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

        if "person_id" in df.columns:
            df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        return df.dropna(subset=["frame", "person_id"])
    except Exception as e:
        st.error(f"Error reading {LOG_FILE}: {e}")
        return pd.DataFrame()


# -------- Main app --------
def main():
    st.set_page_config(
        page_title="Face Detection Dashboard",
        page_icon="📊",
        layout="wide",
    )

    df = load_data()

    st.title("Face Detection Dashboard")
    st.caption(f"Data source: {LOG_FILE}")

    if df.empty:
        st.warning("No data yet. Run DetectionMain.py and come back once detections are logged.")
        return

    # -------- Sidebar filters --------
    st.sidebar.header("Filters")

    # Person ID filter
    person_ids = sorted(df["person_id"].unique())
    default_ids = person_ids  # select all by default

    selected_ids = st.sidebar.multiselect(
        "Person ID(s) to include",
        options=person_ids,
        default=default_ids,
    )

    if not selected_ids:
        st.sidebar.info("Select at least one person ID to show data.")
        st.stop()

    # Frame range filter
    min_frame = int(df["frame"].min())
    max_frame = int(df["frame"].max())

    frame_range = st.sidebar.slider(
        "Frame range",
        min_value=min_frame,
        max_value=max_frame,
        value=(min_frame, max_frame),
    )

    # Apply filters
    mask = df["person_id"].isin(selected_ids) & df["frame"].between(
        frame_range[0], frame_range[1]
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No detections match the current filters.")
        return

    # -------- Summary cards --------
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total detections", len(filtered))

    with col2:
        st.metric("Unique people", int(filtered["person_id"].nunique()))

    with col3:
        frame_min = int(filtered["frame"].min())
        frame_max = int(filtered["frame"].max())
        st.metric("Frame span", f"{frame_min} – {frame_max}")

    # -------- Detections per frame --------
    st.subheader("Detections per frame")
    det_per_frame = (
        filtered.groupby("frame")
        .size()
        .reset_index(name="detections")
        .sort_values("frame")
    )

    st.line_chart(
        det_per_frame.set_index("frame")["detections"],
        height=250,
    )

    # -------- Demographics section --------
    st.subheader("Demographics (buckets)")

    col_a, col_b, col_c = st.columns(3)

    # Gender distribution
    with col_a:
        st.markdown("**Gender distribution**")
        gender_counts = filtered["gender"].fillna("Unknown").value_counts()
        if not gender_counts.empty:
            st.bar_chart(gender_counts)
        else:
            st.info("No gender data available.")

    # Age bucket distribution
    with col_b:
        st.markdown("**Age bucket distribution**")
        # Age strings like '(25-32)' are already buckets
        age_counts = filtered["age"].fillna("Unknown").value_counts()

        # Optional: order by your known buckets so it looks nicer
        age_order = [
            "(0-2)", "(4-6)", "(8-12)", "(15-20)",
            "(25-32)", "(38-43)", "(48-53)", "(60-100)", "Unknown",
        ]
        age_counts = age_counts.reindex(
            [a for a in age_order if a in age_counts.index]
        )

        if not age_counts.empty:
            st.bar_chart(age_counts)
        else:
            st.info("No age data available.")

    # Ethnicity distribution
    with col_c:
        st.markdown("**Ethnicity distribution**")
        eth_counts = filtered["ethnicity"].fillna("Unknown").value_counts()
        if not eth_counts.empty:
            st.bar_chart(eth_counts)
        else:
            st.info("No ethnicity data available.")

    # -------- Raw data preview --------
    st.subheader("Raw detections (filtered)")
    st.dataframe(
        filtered.sort_values(["frame", "person_id"]).reset_index(drop=True),
        use_container_width=True,
        height=300,
    )


if __name__ == "__main__":
    main()


    # ============================================================
    # 🔁 RESET & RESTART CHEAT-SHEET (FOR TERMINAL)
    # ============================================================
    #
    # ✅ STEP 1 — GO TO THE CORRECT FOLDER
    # ------------------------------------------------------------
    # cd C:\Users\wilso\Documents\GitHub\CS124HFA2025\opencv_setup\py_scripts
    #
    #
    # ✅ STEP 2 — DELETE ALL OLD DATA (FULL RESET)
    # ------------------------------------------------------------
    # cd logs
    # del detections.csv
    # cd ..
    #
    #
    # ✅ STEP 3 — START DETECTION (DATA GENERATOR)
    # ------------------------------------------------------------
    # python DetectionMain.py
    #
    # Then choose:
    # 1   (for live camera)
    #
    #
    # ✅ STEP 4 — START DASHBOARD (IN A NEW TERMINAL)
    # ------------------------------------------------------------
    # cd C:\Users\wilso\Documents\GitHub\CS124HFA2025\opencv_setup\py_scripts
    # streamlit run dashboard_app.py
    #
    # If Streamlit is not recognized:
    # python -m streamlit run dashboard_app.py
    #
    #
    # ✅ STEP 5 — STOP EVERYTHING
    # ------------------------------------------------------------
    # In BOTH terminals press:
    # Ctrl + C
    #
    # ============================================================