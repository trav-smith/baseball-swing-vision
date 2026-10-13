
from pathlib import Path
import tempfile
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# MODEL SETTINGS
# ============================================================

POSE_MODEL_NAME = "yolo11n-pose.pt"
SEG_MODEL_NAME = "yolo11s-seg.pt"
BAT_CLASS_ID = 34
BALL_CLASS_ID = 32

WRIST_CONF_THRESHOLD = 0.60
BAT_DISTANCE_THRESHOLD = 90

BALL_CONF_THRESHOLD = 0.05
BALL_DEPARTURE_THRESHOLD_PX = 50.0

# Ball track must look like a ball resting on a tee before
# we trust a departure event.
BALL_STABLE_MAX_STEP_PX = 8.0
BALL_MIN_STABLE_FRAMES = 4

# Contact / ball departure must occur across consecutive
# or nearly consecutive detections. Large detection gaps
# cannot be interpreted as ball flight.
BALL_MAX_CONTACT_FRAME_GAP = 2

# If the same "ball" remains essentially stationary for a
# long period, it is likely a false positive / static object.
BALL_STATIC_FALSE_POSITIVE_FRAMES = 12

# MVP performance optimization:
# only search for the ball/contact event in the middle
# portion of the clip rather than on every frame.
BALL_SEARCH_START_FRACTION = 0.20
BALL_SEARCH_END_FRACTION = 0.85

_pose_model = None
_seg_model = None


def get_pose_model():
    global _pose_model

    if _pose_model is None:
        _pose_model = YOLO(
            POSE_MODEL_NAME
        )

    return _pose_model


def get_seg_model():
    global _seg_model

    if _seg_model is None:
        _seg_model = YOLO(
            SEG_MODEL_NAME
        )

    return _seg_model



# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_swing(video_path, handedness):

    video_path = Path(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration_sec = (
        frame_count / fps
        if fps > 0
        else None
    )

    ball_search_start_frame = int(
        frame_count
        * BALL_SEARCH_START_FRACTION
    )

    ball_search_end_frame = int(
        frame_count
        * BALL_SEARCH_END_FRACTION
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    raw_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    raw_video_path = Path(
        raw_file.name
    )

    raw_file.close()

    final_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    annotated_video_path = Path(
        final_file.name
    )

    final_file.close()

    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    # ========================================================
    # MODELS
    # ========================================================

    pose_model = get_pose_model()
    seg_model = get_seg_model()

    # ========================================================
    # COUNTERS
    # ========================================================

    processed_frames = 0

    pose_detected_frames = 0
    bat_tracked_frames = 0
    ball_detected_frames = 0

    # Frame-by-frame barrel endpoint data
    barrel_tip_rows = []

    # Trusted bat masks used for the swing-path keyframe strip.
    # Only accepted wrist-gated masks are stored.
    bat_mask_by_frame = {}

    # Torso scale for camera-distance normalization
    torso_scale_by_frame = {}

    # Reliable 2D head center by frame
    head_center_by_frame = {}

    # Reliable wrist / hand center by frame.
    # Used only to determine WHEN the active swing occurs.
    hand_center_by_frame = {}

    # ========================================================
    # CONTACT STATE
    # ========================================================

    previous_ball = None
    stable_ball_run = []
    all_ball_detections = []

    contact_detected = False
    contact_previous_frame = None
    contact_frame = None

    contact_previous_time = None
    contact_time = None

    contact_displacement_px = None

    # Keep contact banner visible briefly
    contact_banner_frames = max(
        1,
        int(round(fps * 0.80))
    )

    contact_banner_until = None

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        processed_frames += 1

        annotated = frame.copy()

        # ====================================================
        # 1. POSE
        # ====================================================

        pose_result = pose_model.predict(
            frame,
            verbose=False
        )[0]

        reliable_wrists = []

        if (
            pose_result.keypoints is not None
            and pose_result.keypoints.xy is not None
            and len(pose_result.keypoints.xy) > 0
        ):

            pose_detected_frames += 1

            xy = (
                pose_result.keypoints.xy[0]
                .cpu()
                .numpy()
            )

            conf = (
                pose_result.keypoints.conf[0]
                .cpu()
                .numpy()
            )

            # ----------------------------------------
            # Head center
            # ----------------------------------------

            head_points = []

            for head_idx in [0, 1, 2, 3, 4]:

                if (
                    head_idx < len(conf)
                    and conf[head_idx] >= 0.50
                ):

                    head_points.append(
                        xy[head_idx]
                    )

            if len(head_points) >= 2:

                head_center = np.mean(
                    np.array(
                        head_points
                    ),
                    axis=0
                )

                head_center_by_frame[
                    frame_index
                ] = {
                    "x": float(
                        head_center[0]
                    ),
                    "y": float(
                        head_center[1]
                    )
                }

            # ----------------------------------------
            # Torso scale
            # ----------------------------------------

            torso_indices = [5, 6, 11, 12]

            if all(
                idx < len(conf)
                and conf[idx] >= 0.50
                for idx in torso_indices
            ):

                shoulder_mid = (
                    xy[5] + xy[6]
                ) / 2.0

                hip_mid = (
                    xy[11] + xy[12]
                ) / 2.0

                torso_length = float(
                    np.linalg.norm(
                        shoulder_mid - hip_mid
                    )
                )

                if torso_length > 1:
                    torso_scale_by_frame[
                        frame_index
                    ] = torso_length

            # COCO:
            # 9 = left wrist
            # 10 = right wrist

            for wrist_idx in [9, 10]:

                if (
                    wrist_idx < len(conf)
                    and conf[wrist_idx]
                    >= WRIST_CONF_THRESHOLD
                ):

                    reliable_wrists.append(
                        np.array([
                            float(
                                xy[wrist_idx][0]
                            ),
                            float(
                                xy[wrist_idx][1]
                            )
                        ])
                    )

            # Existing Ultralytics pose overlay
            annotated = pose_result.plot()

        # ----------------------------------------------------
        # Store hand center for swing-timing detection
        # ----------------------------------------------------

        if reliable_wrists:

            hand_center = np.mean(
                np.array(
                    reliable_wrists
                ),
                axis=0
            )

            hand_center_by_frame[
                frame_index
            ] = {
                "x": float(
                    hand_center[0]
                ),
                "y": float(
                    hand_center[1]
                )
            }

        # ====================================================
        # 2. BAT SEGMENTATION
        # ====================================================

        bat_tracked = False

        if reliable_wrists:

            seg_classes = [BAT_CLASS_ID]

            if (
                frame_index >= ball_search_start_frame
                and frame_index <= ball_search_end_frame
            ):
                seg_classes.append(
                    BALL_CLASS_ID
                )

            seg_result = seg_model.predict(
                frame,
                classes=seg_classes,
                conf=0.05,
                verbose=False
            )[0]

            if (
                seg_result.masks is not None
                and seg_result.boxes is not None
                and len(seg_result.boxes) > 0
            ):

                masks = (
                    seg_result.masks.data
                    .cpu()
                    .numpy()
                )

                seg_classes_found = (
                    seg_result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                best_mask = None
                best_distance = None

                for mask, class_id in zip(
                    masks,
                    seg_classes_found
                ):

                    if class_id != BAT_CLASS_ID:
                        continue

                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                    mask_binary = (
                        mask > 0.5
                    )

                    ys, xs = np.where(
                        mask_binary
                    )

                    if len(xs) == 0:
                        continue

                    nearest_distance = None

                    for wrist_xy in reliable_wrists:

                        distances = np.sqrt(
                            (xs - wrist_xy[0])**2
                            +
                            (ys - wrist_xy[1])**2
                        )

                        distance = float(
                            distances.min()
                        )

                        if (
                            nearest_distance is None
                            or distance
                            < nearest_distance
                        ):
                            nearest_distance = (
                                distance
                            )

                    if (
                        best_distance is None
                        or nearest_distance
                        < best_distance
                    ):

                        best_distance = (
                            nearest_distance
                        )

                        best_mask = (
                            mask_binary.copy()
                        )

                if (
                    best_mask is not None
                    and best_distance
                    <= BAT_DISTANCE_THRESHOLD
                ):

                    bat_tracked = True
                    bat_tracked_frames += 1

                    bat_mask_by_frame[
                        frame_index
                    ] = best_mask.copy()

                    overlay = (
                        annotated.copy()
                    )

                    overlay[
                        best_mask
                    ] = (60, 220, 80)

                    annotated = (
                        cv2.addWeighted(
                            annotated,
                            0.72,
                            overlay,
                            0.28,
                            0
                        )
                    )

                    # ========================================
                    # BARREL-ENDPOINT FROM BAT MASK
                    # ========================================

                    # ========================================
                    # BARREL ENDPOINT FROM ALL TRUSTED MASK PIXELS
                    # ========================================

                    mask_y, mask_x = np.where(
                        best_mask
                    )

                    if len(mask_x) >= 10:

                        points = np.column_stack(
                            (
                                mask_x,
                                mask_y
                            )
                        ).astype(np.float32)

                        # -------------------------------
                        # Find nearest reliable wrist
                        # -------------------------------

                        nearest_wrist = None
                        nearest_wrist_distance = None

                        for wrist_xy in reliable_wrists:

                            distances = np.sqrt(
                                (
                                    points[:, 0]
                                    - wrist_xy[0]
                                )**2
                                +
                                (
                                    points[:, 1]
                                    - wrist_xy[1]
                                )**2
                            )

                            wrist_distance = float(
                                distances.min()
                            )

                            if (
                                nearest_wrist_distance is None
                                or wrist_distance
                                < nearest_wrist_distance
                            ):

                                nearest_wrist_distance = (
                                    wrist_distance
                                )

                                nearest_wrist = (
                                    wrist_xy
                                )

                        # -------------------------------
                        # PCA long axis
                        # -------------------------------

                        mean = np.mean(
                            points,
                            axis=0
                        )

                        centered = (
                            points - mean
                        )

                        covariance = np.cov(
                            centered,
                            rowvar=False
                        )

                        eigenvalues, eigenvectors = (
                            np.linalg.eigh(
                                covariance
                            )
                        )

                        long_axis = (
                            eigenvectors[
                                :,
                                np.argmax(
                                    eigenvalues
                                )
                            ]
                        )

                        projections = (
                            centered
                            @ long_axis
                        )

                        endpoint_1 = (
                            mean
                            +
                            long_axis
                            * projections.min()
                        )

                        endpoint_2 = (
                            mean
                            +
                            long_axis
                            * projections.max()
                        )

                        # -------------------------------
                        # Barrel = endpoint farther
                        # from nearest wrist
                        # -------------------------------

                        if nearest_wrist is not None:

                            d1 = float(
                                np.linalg.norm(
                                    endpoint_1
                                    - nearest_wrist
                                )
                            )

                            d2 = float(
                                np.linalg.norm(
                                    endpoint_2
                                    - nearest_wrist
                                )
                            )

                            barrel_tip = (
                                endpoint_1
                                if d1 > d2
                                else endpoint_2
                            )

                            tip_x = float(
                                barrel_tip[0]
                            )

                            tip_y = float(
                                barrel_tip[1]
                            )

                            apparent_length_px = float(
                                np.linalg.norm(
                                    endpoint_1
                                    - endpoint_2
                                )
                            )

                            barrel_tip_rows.append({
                                "frame":
                                    frame_index,

                                "time_sec":
                                    frame_index / fps,

                                "x":
                                    tip_x,

                                "y":
                                    tip_y,

                                "apparent_length_px":
                                    apparent_length_px,

                                "wrist_distance_px":
                                    float(
                                        nearest_wrist_distance
                                    )
                            })

                            # Visible validation marker
                            cv2.circle(
                                annotated,
                                (
                                    int(tip_x),
                                    int(tip_y)
                                ),
                                8,
                                (255, 0, 255),
                                -1,
                                cv2.LINE_AA
                            )

                            cv2.circle(
                                annotated,
                                (
                                    int(tip_x),
                                    int(tip_y)
                                ),
                                12,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

        # ====================================================
        # 3. BALL DETECTION
        # ====================================================

        current_ball = None

        # Reuse the segmentation-model inference that was
        # already run for bat tracking on this frame.
        if (
            reliable_wrists
            and frame_index >= ball_search_start_frame
            and frame_index <= ball_search_end_frame
            and seg_result.boxes is not None
            and len(seg_result.boxes) > 0
        ):

            boxes = (
                seg_result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                seg_result.boxes.conf
                .cpu()
                .numpy()
            )

            classes_found = (
                seg_result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            ball_indices = np.where(
                classes_found == BALL_CLASS_ID
            )[0]

            if len(ball_indices) > 0:

                best_idx = int(
                    ball_indices[
                        np.argmax(
                            confidences[
                                ball_indices
                            ]
                        )
                    ]
                )

                x1, y1, x2, y2 = (
                    boxes[best_idx]
                )

                ball_conf = float(
                    confidences[
                        best_idx
                    ]
                )

            else:
                best_idx = None

        else:
            best_idx = None

        if best_idx is not None:

            ball_x = float(
                (x1 + x2) / 2
            )

            ball_y = float(
                (y1 + y2) / 2
            )

            current_ball = {
                "frame": frame_index,
                "x": ball_x,
                "y": ball_y,
                "confidence": ball_conf
            }

            all_ball_detections.append(
                current_ball.copy()
            )

            ball_detected_frames += 1

            # -----------------------------------------------
            # Draw ball marker
            # -----------------------------------------------

            cv2.circle(
                annotated,
                (
                    int(ball_x),
                    int(ball_y)
                ),
                12,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.circle(
                annotated,
                (
                    int(ball_x),
                    int(ball_y)
                ),
                5,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )

        # ====================================================
        # 4. BALL DEPARTURE / CONTACT INTERVAL
        # ====================================================

        if (
            previous_ball is not None
            and current_ball is not None
        ):

            dx = (
                current_ball["x"]
                -
                previous_ball["x"]
            )

            dy = (
                current_ball["y"]
                -
                previous_ball["y"]
            )

            displacement = float(
                np.sqrt(
                    dx**2 + dy**2
                )
            )

            # Build a stable pre-contact track.
            if (
                displacement
                <= BALL_STABLE_MAX_STEP_PX
            ):

                if not stable_ball_run:
                    stable_ball_run.append(
                        previous_ball.copy()
                    )

                stable_ball_run.append(
                    current_ball.copy()
                )

            else:

                # Only call this contact if the ball was
                # genuinely stable for several detections
                # immediately beforehand.
                frame_gap = (
                    current_ball["frame"]
                    - previous_ball["frame"]
                )

                if (
                    not contact_detected
                    and len(stable_ball_run)
                    >= BALL_MIN_STABLE_FRAMES
                    and frame_gap
                    <= BALL_MAX_CONTACT_FRAME_GAP
                    and displacement
                    >= BALL_DEPARTURE_THRESHOLD_PX
                ):

                    contact_detected = True

                    contact_previous_frame = int(
                        previous_ball["frame"]
                    )

                    contact_frame = int(
                        current_ball["frame"]
                    )

                    contact_previous_time = (
                        contact_previous_frame
                        / fps
                    )

                    contact_time = (
                        contact_frame
                        / fps
                    )

                    contact_displacement_px = (
                        displacement
                    )

                    contact_banner_until = (
                        contact_frame
                        +
                        contact_banner_frames
                    )

                # New object / broken track:
                # do not stitch unrelated detections.
                stable_ball_run = []

        # Only advance previous-ball state when
        # a ball was actually detected.
        if current_ball is not None:
            previous_ball = current_ball

        # ====================================================
        # 5. STATUS OVERLAYS
        # ====================================================

        if bat_tracked:

            cv2.putText(
                annotated,
                "BAT TRACKED",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (60, 220, 80),
                2,
                cv2.LINE_AA
            )

        # Contact banner
        if (
            contact_banner_until
            is not None
            and frame_index
            <= contact_banner_until
            and frame_index
            >= contact_frame
        ):

            banner_text = (
                f"CONTACT DETECTED  "
                f"{contact_previous_time:.3f}-"
                f"{contact_time:.3f} sec"
            )

            text_size = cv2.getTextSize(
                banner_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                2
            )[0]

            text_x = int(
                (
                    width
                    -
                    text_size[0]
                )
                / 2
            )

            cv2.rectangle(
                annotated,
                (
                    text_x - 18,
                    18
                ),
                (
                    text_x
                    + text_size[0]
                    + 18,
                    62
                ),
                (15, 15, 15),
                -1
            )

            cv2.putText(
                annotated,
                banner_text,
                (
                    text_x,
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        writer.write(
            annotated
        )

        frame_index += 1

    cap.release()
    writer.release()

    # ========================================================
    # H.264 BROWSER OUTPUT
    # ========================================================

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(annotated_video_path)
    ]

    subprocess.run(
        ffmpeg_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # ========================================================
    # COVERAGE METRICS
    # ========================================================

    pose_coverage = (
        pose_detected_frames
        / processed_frames
        if processed_frames > 0
        else 0
    )

    bat_coverage = (
        bat_tracked_frames
        / processed_frames
        if processed_frames > 0
        else 0
    )

    ball_coverage = (
        ball_detected_frames
        / processed_frames
        if processed_frames > 0
        else 0
    )

    # ========================================================
    # ACTIVE SWING WINDOW FROM WRIST MOTION
    # ========================================================

    swing_window_start = None
    swing_window_end = None
    wrist_peak_frame = None

    wrist_speed_rows = []

    if len(
        hand_center_by_frame
    ) >= 3:

        for start_frame_num in sorted(
            hand_center_by_frame.keys()
        ):

            end_frame_num = (
                start_frame_num + 2
            )

            if (
                end_frame_num
                not in hand_center_by_frame
            ):
                continue

            p1 = hand_center_by_frame[
                start_frame_num
            ]

            p2 = hand_center_by_frame[
                end_frame_num
            ]

            distance_px = float(
                np.sqrt(
                    (p2["x"] - p1["x"])**2
                    +
                    (p2["y"] - p1["y"])**2
                )
            )

            speed_px_sec = (
                distance_px
                /
                (2.0 / fps)
            )

            wrist_speed_rows.append({
                "frame":
                    end_frame_num,

                "speed_px_sec":
                    speed_px_sec
            })

    if wrist_speed_rows:

        wrist_speed_rows = sorted(
            wrist_speed_rows,
            key=lambda row:
                row["frame"]
        )

        wrist_speeds = np.array(
            [
                row["speed_px_sec"]
                for row
                in wrist_speed_rows
            ],
            dtype=float
        )

        peak_idx = int(
            np.argmax(
                wrist_speeds
            )
        )

        wrist_peak_frame = int(
            wrist_speed_rows[
                peak_idx
            ]["frame"]
        )

        wrist_peak_speed = float(
            wrist_speed_rows[
                peak_idx
            ]["speed_px_sec"]
        )

        # -----------------------------------------------
        # Find swing initiation
        # -----------------------------------------------

        onset_threshold = (
            wrist_peak_speed
            * 0.18
        )

        start_idx = 0
        low_run = 0

        for i in range(
            peak_idx - 1,
            -1,
            -1
        ):

            if (
                wrist_speed_rows[i][
                    "speed_px_sec"
                ]
                < onset_threshold
            ):

                low_run += 1

            else:

                low_run = 0

            if low_run >= 3:

                start_idx = min(
                    i + 3,
                    peak_idx
                )

                break

        swing_window_start = int(
            wrist_speed_rows[
                start_idx
            ]["frame"]
        )

        # Include a small amount of pre-motion context.
        swing_window_start = max(
            0,
            swing_window_start
            - int(
                round(
                    0.08 * fps
                )
            )
        )

        # -----------------------------------------------
        # End shortly after primary hand-speed peak.
        # Avoid later follow-through/recovery peaks.
        # -----------------------------------------------

        swing_window_end = min(
            frame_count - 1,
            wrist_peak_frame
            + int(
                round(
                    0.16 * fps
                )
            )
        )

    # ========================================================
    # RELATIVE 2D BAT SPEED
    # ========================================================

    relative_speed_values = []

    if len(barrel_tip_rows) >= 3:

        barrel_by_frame = {
            int(row["frame"]): row
            for row in barrel_tip_rows
        }

        # Use a 3-frame span:
        # position(frame) -> position(frame + 2)
        # This is more stable than consecutive-frame speed.
        for start_frame_num in sorted(
            barrel_by_frame.keys()
        ):

            end_frame_num = (
                start_frame_num + 2
            )

            if (
                end_frame_num
                not in barrel_by_frame
            ):
                continue

            # Bat speed is only meaningful inside the
            # wrist-defined active swing window.
            if (
                swing_window_start is None
                or swing_window_end is None
                or start_frame_num
                < swing_window_start
                or end_frame_num
                > swing_window_end
            ):
                continue

            if (
                start_frame_num
                not in torso_scale_by_frame
                or end_frame_num
                not in torso_scale_by_frame
            ):
                continue

            p1 = barrel_by_frame[
                start_frame_num
            ]

            p2 = barrel_by_frame[
                end_frame_num
            ]

            distance_px = float(
                np.sqrt(
                    (p2["x"] - p1["x"])**2
                    +
                    (p2["y"] - p1["y"])**2
                )
            )

            delta_sec = (
                2.0 / fps
            )

            speed_px_sec = (
                distance_px
                / delta_sec
            )

            torso_scale = float(
                np.median([
                    torso_scale_by_frame[
                        start_frame_num
                    ],
                    torso_scale_by_frame[
                        end_frame_num
                    ]
                ])
            )

            if torso_scale <= 0:
                continue

            relative_speed = (
                speed_px_sec
                / torso_scale
            )

            # Reject obviously broken endpoint jumps.
            # This is intentionally generous.
            if (
                relative_speed > 0
                and relative_speed < 120
            ):

                relative_speed_values.append(
                    {
                        "frame":
                            end_frame_num,

                        "time_sec":
                            end_frame_num
                            / fps,

                        "relative_speed":
                            float(
                                relative_speed
                            )
                    }
                )

    if relative_speed_values:

        speed_numbers = np.array(
            [
                row["relative_speed"]
                for row
                in relative_speed_values
            ],
            dtype=float
        )

        # Robust peak:
        # median of the three highest valid measurements
        # rather than trusting one isolated maximum.
        top_n = min(
            3,
            len(speed_numbers)
        )

        top_values = np.sort(
            speed_numbers
        )[-top_n:]

        robust_peak_relative_speed = float(
            np.median(
                top_values
            )
        )

        relative_speed_result = {
            "available": True,

            "peak_relative_2d_speed":
                round(
                    robust_peak_relative_speed,
                    1
                ),

            "unit":
                "torso lengths/sec",

            "valid_measurements":
                len(
                    relative_speed_values
                ),

            "note": (
                "Relative 2D camera-based speed; "
                "not radar-gun mph."
            )
        }

    else:

        relative_speed_result = {
            "available": False,

            "reason": (
                "Bat speed could not be measured "
                "reliably from this video."
            )
        }

    # ========================================================
    # HEAD STABILITY
    # ========================================================

    head_stability_result = {
        "available": False,
        "reason": (
            "Head stability could not be measured "
            "reliably from this video."
        )
    }

    # Use meaningful barrel motion to define the active swing.
    if relative_speed_values:

        speed_numbers = np.array(
            [
                row["relative_speed"]
                for row
                in relative_speed_values
            ],
            dtype=float
        )

        peak_for_window = float(
            np.max(
                speed_numbers
            )
        )

        motion_threshold = (
            peak_for_window
            * 0.20
        )

        active_speed_rows = [
            row
            for row
            in relative_speed_values
            if row["relative_speed"]
            >= motion_threshold
        ]

        if len(active_speed_rows) >= 2:

            active_start_frame = int(
                min(
                    row["frame"]
                    for row
                    in active_speed_rows
                )
            )

            active_end_frame = int(
                max(
                    row["frame"]
                    for row
                    in active_speed_rows
                )
            )

            head_frames = [
                frame
                for frame
                in sorted(
                    head_center_by_frame.keys()
                )
                if (
                    frame >= active_start_frame
                    and frame <= active_end_frame
                    and frame in torso_scale_by_frame
                )
            ]

            if len(head_frames) >= 4:

                start_frame = (
                    head_frames[0]
                )

                start_head = np.array([
                    head_center_by_frame[
                        start_frame
                    ]["x"],
                    head_center_by_frame[
                        start_frame
                    ]["y"]
                ])

                normalized_excursions = []

                for frame in head_frames:

                    current_head = np.array([
                        head_center_by_frame[
                            frame
                        ]["x"],
                        head_center_by_frame[
                            frame
                        ]["y"]
                    ])

                    displacement_px = float(
                        np.linalg.norm(
                            current_head
                            - start_head
                        )
                    )

                    torso_scale = float(
                        torso_scale_by_frame[
                            frame
                        ]
                    )

                    if torso_scale > 0:

                        normalized_excursions.append(
                            displacement_px
                            / torso_scale
                        )

                if normalized_excursions:

                    max_excursion = float(
                        np.max(
                            normalized_excursions
                        )
                    )

                    head_stability_result = {
                        "available": True,

                        "max_head_excursion":
                            round(
                                max_excursion,
                                2
                            ),

                        "unit":
                            "torso lengths",

                        "frames_measured":
                            len(
                                head_frames
                            ),

                        "active_window_start_sec":
                            round(
                                active_start_frame
                                / fps,
                                3
                            ),

                        "active_window_end_sec":
                            round(
                                active_end_frame
                                / fps,
                                3
                            ),

                        "note": (
                            "Maximum 2D head movement "
                            "during the active swing."
                        )
                    }

    # ========================================================
    # SWING-PATH KEYFRAME VISUALIZATION
    # ========================================================

    swing_path_visual_path = None

    if (
        relative_speed_values
        and len(bat_mask_by_frame) >= 3
    ):

        # ----------------------------------------------------
        # Use the SAME wrist-defined active swing window used
        # for the bat-speed metric.
        #
        # This prevents setup/load barrel jitter from defining
        # the swing and avoids late follow-through frames.
        # ----------------------------------------------------

        if (
            swing_window_start is not None
            and swing_window_end is not None
        ):

            visual_start = (
                swing_window_start
            )

            visual_end = (
                swing_window_end
            )

        else:

            # Fail conservatively if swing timing could not
            # be established from wrist motion.
            visual_start = 0
            visual_end = -1

        # ----------------------------------------------------
        # Barrel continuity quality gate
        #
        # Split the tracked barrel sequence at implausibly large
        # frame-to-frame jumps. This prevents an object-switch
        # from appearing as part of the displayed swing path.
        # ----------------------------------------------------

        raw_candidate_frames = [
            frame_num
            for frame_num
            in sorted(
                bat_mask_by_frame.keys()
            )
            if (
                frame_num >= visual_start
                and frame_num <= visual_end
            )
        ]

        barrel_lookup_for_path = {
            int(row["frame"]): row
            for row in barrel_tip_rows
        }

        CONTINUITY_MAX_JUMP_PER_FRAME_PX = 180.0
        CONTINUITY_MAX_FRAME_GAP = 3

        continuity_segments = []
        current_segment = []

        previous_frame = None

        for frame_num in raw_candidate_frames:

            if frame_num not in barrel_lookup_for_path:
                continue

            if previous_frame is None:

                current_segment = [
                    frame_num
                ]

            else:

                frame_gap = (
                    frame_num
                    - previous_frame
                )

                previous_tip = (
                    barrel_lookup_for_path[
                        previous_frame
                    ]
                )

                current_tip = (
                    barrel_lookup_for_path[
                        frame_num
                    ]
                )

                jump_px = float(
                    np.sqrt(
                        (
                            current_tip["x"]
                            - previous_tip["x"]
                        )**2
                        +
                        (
                            current_tip["y"]
                            - previous_tip["y"]
                        )**2
                    )
                )

                jump_per_frame = (
                    jump_px / frame_gap
                    if frame_gap > 0
                    else float("inf")
                )

                continuity_break = (
                    frame_gap
                    > CONTINUITY_MAX_FRAME_GAP
                    or jump_per_frame
                    > CONTINUITY_MAX_JUMP_PER_FRAME_PX
                )

                if continuity_break:

                    if current_segment:
                        continuity_segments.append(
                            current_segment
                        )

                    current_segment = [
                        frame_num
                    ]

                else:

                    current_segment.append(
                        frame_num
                    )

            previous_frame = frame_num

        if current_segment:

            continuity_segments.append(
                current_segment
            )

        # Prefer the segment containing the primary wrist-speed
        # peak. Otherwise choose the segment whose frames are
        # closest to that peak.
        if continuity_segments:

            containing_peak = [
                segment
                for segment
                in continuity_segments
                if (
                    wrist_peak_frame is not None
                    and min(segment)
                    <= wrist_peak_frame
                    <= max(segment)
                )
            ]

            if containing_peak:

                best_segment = max(
                    containing_peak,
                    key=len
                )

                # A trustworthy swing path must continue
                # beyond the primary hand-speed peak.
                #
                # If barrel tracking breaks at/near peak speed,
                # showing only the load/approach would be
                # misleading.
                MIN_POST_PEAK_SEC = 0.10

                min_post_peak_frames = max(
                    2,
                    int(
                        round(
                            MIN_POST_PEAK_SEC * fps
                        )
                    )
                )

                required_end_frame = (
                    wrist_peak_frame
                    + min_post_peak_frames
                )

                survives_post_peak = (
                    max(best_segment)
                    >= required_end_frame
                )

                # If we have a reliable contact interval,
                # reaching the last stable pre-departure frame
                # is also sufficient.
                #
                # This allows legitimate cases where tracking
                # breaks immediately after contact because of
                # motion blur / bat speed.
                reaches_contact = False

                if (
                    contact_detected
                    and contact_previous_frame is not None
                ):

                    reaches_contact = (
                        max(best_segment)
                        >= contact_previous_frame
                    )

                if (
                    survives_post_peak
                    or reaches_contact
                ):

                    candidate_frames = (
                        best_segment
                    )

                else:

                    candidate_frames = []

            else:

                # No continuous trusted barrel segment spans
                # the primary swing-speed peak.
                candidate_frames = []

        else:

            candidate_frames = []

        # ------------------------------------------------
        # Choose up to 5 actual tracked bat positions
        # ------------------------------------------------

        NUM_KEYFRAMES = 5

        if len(candidate_frames) >= 3:

            if (
                len(candidate_frames)
                <= NUM_KEYFRAMES
            ):

                selected_frames = (
                    candidate_frames
                )

            else:

                selected_indices = (
                    np.linspace(
                        0,
                        len(candidate_frames) - 1,
                        NUM_KEYFRAMES
                    )
                    .round()
                    .astype(int)
                )

                selected_frames = [
                    candidate_frames[i]
                    for i in selected_indices
                ]

            # If contact is reliable and lies inside the
            # swing, include its nearest tracked frame.
            if (
                contact_detected
                and contact_frame is not None
                and candidate_frames
            ):

                nearest_contact_frame = min(
                    candidate_frames,
                    key=lambda f: abs(
                        f - contact_frame
                    )
                )

                if (
                    nearest_contact_frame
                    not in selected_frames
                ):

                    # Replace the selected frame nearest
                    # contact so total stays at ~5.
                    replace_idx = min(
                        range(
                            len(
                                selected_frames
                            )
                        ),
                        key=lambda i: abs(
                            selected_frames[i]
                            - contact_frame
                        )
                    )

                    selected_frames[
                        replace_idx
                    ] = (
                        nearest_contact_frame
                    )

                    selected_frames = sorted(
                        set(
                            selected_frames
                        )
                    )

            # ------------------------------------------------
            # Load actual source frames
            # ------------------------------------------------

            frame_cap = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            keyframe_images = []

            barrel_lookup = {
                int(row["frame"]): row
                for row
                in barrel_tip_rows
            }

            for frame_num in selected_frames:

                frame_cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(
                        frame_num
                    )
                )

                ok_frame, keyframe = (
                    frame_cap.read()
                )

                if not ok_frame:
                    continue

                mask = bat_mask_by_frame.get(
                    frame_num
                )

                if mask is not None:

                    # Green bat overlay
                    mask_overlay = (
                        keyframe.copy()
                    )

                    mask_overlay[
                        mask
                    ] = (
                        60,
                        220,
                        80
                    )

                    keyframe = (
                        cv2.addWeighted(
                            keyframe,
                            0.72,
                            mask_overlay,
                            0.28,
                            0
                        )
                    )

                # Barrel endpoint marker
                barrel_row = (
                    barrel_lookup.get(
                        frame_num
                    )
                )

                if barrel_row is not None:

                    tip_x = int(
                        barrel_row["x"]
                    )

                    tip_y = int(
                        barrel_row["y"]
                    )

                    cv2.circle(
                        keyframe,
                        (
                            tip_x,
                            tip_y
                        ),
                        8,
                        (
                            255,
                            0,
                            255
                        ),
                        -1,
                        cv2.LINE_AA
                    )

                    cv2.circle(
                        keyframe,
                        (
                            tip_x,
                            tip_y
                        ),
                        12,
                        (
                            255,
                            255,
                            255
                        ),
                        2,
                        cv2.LINE_AA
                    )

                # --------------------------------------------
                # Label only contact when reliable
                # --------------------------------------------

                is_contact_panel = (
                    contact_detected
                    and contact_frame
                    is not None
                    and abs(
                        frame_num
                        - contact_frame
                    )
                    <= 1
                )

                if is_contact_panel:

                    cv2.putText(
                        keyframe,
                        "CONTACT",
                        (
                            24,
                            48
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (
                            255,
                            255,
                            255
                        ),
                        3,
                        cv2.LINE_AA
                    )

                keyframe_images.append(
                    (
                        frame_num,
                        keyframe
                    )
                )

            frame_cap.release()

            # ------------------------------------------------
            # Build one horizontal strip
            # ------------------------------------------------

            if keyframe_images:

                PANEL_WIDTH = 320

                resized_panels = []

                for _, img in keyframe_images:

                    panel_height = int(
                        img.shape[0]
                        * (
                            PANEL_WIDTH
                            / img.shape[1]
                        )
                    )

                    panel = cv2.resize(
                        img,
                        (
                            PANEL_WIDTH,
                            panel_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                    resized_panels.append(
                        panel
                    )

                min_height = min(
                    panel.shape[0]
                    for panel
                    in resized_panels
                )

                resized_panels = [
                    cv2.resize(
                        panel,
                        (
                            PANEL_WIDTH,
                            min_height
                        )
                    )
                    for panel
                    in resized_panels
                ]

                strip = cv2.hconcat(
                    resized_panels
                )

                # Header
                HEADER_HEIGHT = 72

                canvas = np.full(
                    (
                        strip.shape[0]
                        + HEADER_HEIGHT,
                        strip.shape[1],
                        3
                    ),
                    20,
                    dtype=np.uint8
                )

                canvas[
                    HEADER_HEIGHT:,
                    :
                ] = strip

                cv2.putText(
                    canvas,
                    "SWING PATH",
                    (
                        22,
                        30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (
                        255,
                        255,
                        255
                    ),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    canvas,
                    (
                        "Tracked bat positions "
                        "through the active swing"
                    ),
                    (
                        22,
                        56
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (
                        190,
                        190,
                        190
                    ),
                    1,
                    cv2.LINE_AA
                )

                # Direction arrow across header
                arrow_start_x = int(
                    strip.shape[1]
                    * 0.62
                )

                arrow_end_x = int(
                    strip.shape[1]
                    * 0.95
                )

                cv2.arrowedLine(
                    canvas,
                    (
                        arrow_start_x,
                        36
                    ),
                    (
                        arrow_end_x,
                        36
                    ),
                    (
                        220,
                        220,
                        220
                    ),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.04
                )

                visual_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".jpg"
                    )
                )

                swing_path_visual_path = (
                    visual_file.name
                )

                visual_file.close()

                cv2.imwrite(
                    swing_path_visual_path,
                    canvas,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        92
                    ]
                )

# ========================================================
    # CONTACT RESULT
    # ========================================================

    if contact_detected:

        contact_result = {
            "detected": True,

            "previous_frame":
                contact_previous_frame,

            "departure_frame":
                contact_frame,

            "start_sec": round(
                contact_previous_time,
                3
            ),

            "end_sec": round(
                contact_time,
                3
            ),

            "displacement_px": round(
                contact_displacement_px,
                1
            )
        }

    else:

        contact_result = {
            "detected": False,

            "reason": (
                "No reliable ball-departure "
                "event detected."
            )
        }

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "status": "complete",

        "handedness": handedness,

        "video": {
            "filename":
                video_path.name,

            "fps":
                round(fps, 2),

            "frame_count":
                frame_count,

            "width":
                width,

            "height":
                height,

            "duration_sec": (
                round(
                    duration_sec,
                    2
                )
                if duration_sec
                is not None
                else None
            )
        },

        "pose": {
            "frames_processed":
                processed_frames,

            "frames_detected":
                pose_detected_frames,

            "coverage":
                round(
                    pose_coverage,
                    3
                )
        },

        "bat": {
            "frames_tracked":
                bat_tracked_frames,

            "coverage":
                round(
                    bat_coverage,
                    3
                )
        },

        "barrel": {
            "frames_tracked":
                len(
                    barrel_tip_rows
                ),

            "coverage":
                round(
                    (
                        len(barrel_tip_rows)
                        / processed_frames
                    )
                    if processed_frames > 0
                    else 0,
                    3
                ),

            "track":
                barrel_tip_rows
        },

        "ball": {
            "frames_detected":
                ball_detected_frames,

            "coverage":
                round(
                    ball_coverage,
                    3
                )
        },

        "contact":
            contact_result,

        "bat_speed":
            relative_speed_result,

        "swing_timing": {
            "available":
                swing_window_start
                is not None,

            "start_frame":
                swing_window_start,

            "end_frame":
                swing_window_end,

            "peak_wrist_frame":
                wrist_peak_frame,

            "start_sec": (
                round(
                    swing_window_start
                    / fps,
                    3
                )
                if swing_window_start
                is not None
                else None
            ),

            "end_sec": (
                round(
                    swing_window_end
                    / fps,
                    3
                )
                if swing_window_end
                is not None
                else None
            )
        },

        "head_stability":
            head_stability_result,

        "swing_path_visual_path":
            swing_path_visual_path,

        "annotated_video_path":
            str(
                annotated_video_path
            ),

        "message": (
            "Swing analysis completed "
            "successfully."
        )
    }
