
import tempfile
from pathlib import Path

import streamlit as st

from swing_analysis import analyze_swing


st.set_page_config(
    page_title="Baseball Swing Vision",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ Baseball Swing Vision")

st.caption(
    "Upload one baseball swing video and generate "
    "computer-vision-based swing analysis."
)

uploaded_video = st.file_uploader(
    "Upload swing video",
    type=["mp4", "mov", "m4v"]
)

handedness_label = st.radio(
    "Batter handedness",
    options=["Right-handed", "Left-handed"],
    horizontal=True
)

analyze_button = st.button(
    "Analyze Swing",
    type="primary",
    disabled=(uploaded_video is None)
)

if uploaded_video is not None:

    st.subheader("Uploaded Swing")

    st.video(
        uploaded_video
    )

if analyze_button and uploaded_video is not None:

    handedness = (
        "R"
        if handedness_label == "Right-handed"
        else "L"
    )

    suffix = Path(
        uploaded_video.name
    ).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp_file:

        tmp_file.write(
            uploaded_video.getbuffer()
        )

        temp_video_path = tmp_file.name

    with st.spinner(
        "Analyzing swing..."
    ):

        result = analyze_swing(
            temp_video_path,
            handedness
        )

    st.success(
        result["message"]
    )


    # --------------------------------------------------------
    # Result payloads
    # --------------------------------------------------------

    pose = result["pose"]
    bat = result["bat"]
    barrel = result.get("barrel", {})
    bat_speed = result.get("bat_speed", {})
    head_stability = result.get("head_stability", {})
    swing_path_visual_path = result.get(
        "swing_path_visual_path"
    )
    contact = result["contact"]
    video = result["video"]

    annotated_video_path = result[
        "annotated_video_path"
    ]

    # --------------------------------------------------------
    # Athlete-facing summary
    # --------------------------------------------------------

    st.header(
        "Swing Analysis"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.subheader(
            "Bat Speed"
        )

        if bat_speed.get(
            "available",
            False
        ):

            st.metric(
                "Relative 2D Bat Speed",
                (
                    f'{bat_speed["peak_relative_2d_speed"]:.1f} '
                    f'{bat_speed["unit"]}'
                )
            )

            st.caption(
                bat_speed.get(
                    "note",
                    "Relative 2D camera-based speed."
                )
            )

        else:

            st.info(
                bat_speed.get(
                    "reason",
                    "Bat speed could not be measured reliably."
                )
            )

    with metric_col2:

        st.subheader(
            "Head Stability"
        )

        if head_stability.get(
            "available",
            False
        ):

            st.metric(
                "Max 2D Head Movement",
                (
                    f'{head_stability["max_head_excursion"]:.2f} '
                    f'{head_stability["unit"]}'
                )
            )

            st.caption(
                head_stability.get(
                    "note",
                    "Maximum 2D head movement during the active swing."
                )
            )

        else:

            st.info(
                head_stability.get(
                    "reason",
                    "Head stability could not be measured reliably."
                )
            )

    # --------------------------------------------------------
    # Swing path
    # --------------------------------------------------------

    st.subheader(
        "Swing Path"
    )

    if swing_path_visual_path:

        st.image(
            swing_path_visual_path,
            caption=(
                "Tracked bat positions through the active swing"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Swing path could not be visualized reliably "
            "from this video."
        )

    # --------------------------------------------------------
    # Annotated swing video
    # --------------------------------------------------------

    st.subheader(
        "Computer Vision Replay"
    )

    with open(
        annotated_video_path,
        "rb"
    ) as video_file:

        annotated_video_bytes = (
            video_file.read()
        )

    st.video(
        annotated_video_bytes,
        format="video/mp4"
    )

    # --------------------------------------------------------
    # Contact result
    # --------------------------------------------------------

    if contact.get(
        "detected",
        False
    ):

        st.subheader(
            "Contact Detection"
        )

        st.success(
            "Ball departure detected"
        )

        st.write(
            f'**Detected contact window:** '
            f'{contact["start_sec"]:.3f}–'
            f'{contact["end_sec"]:.3f} sec'
        )

    # --------------------------------------------------------
    # Analysis quality
    # --------------------------------------------------------

    with st.expander(
        "Analysis Quality"
    ):

        quality_col1, quality_col2, quality_col3 = (
            st.columns(3)
        )

        quality_col1.metric(
            "Pose Tracking",
            f'{pose["coverage"] * 100:.1f}%'
        )

        quality_col2.metric(
            "Bat Tracking",
            f'{bat["coverage"] * 100:.1f}%'
        )

        quality_col3.metric(
            "Barrel Tracking",
            (
                f'{barrel.get("coverage", 0) * 100:.1f}%'
            )
        )

        st.caption(
            (
                f'Barrel endpoint tracked on '
                f'{barrel.get("frames_tracked", 0)} frames. '
                f'Clip duration: '
                f'{video["duration_sec"]:.2f} sec.'
            )
        )

        if not contact.get(
            "detected",
            False
        ):

            st.info(
                "Contact could not be measured reliably "
                "from this video."
            )

        st.caption(
            "Measurements are based on 2D video. "
            "Unreliable outputs are suppressed rather than guessed."
        )
