"""
AHP Task Prioritization (Streamlit + streamlit-tags)
---------------------------------------------------
Run:
    streamlit run app_ahp_task_prioritization_tags.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags

st.set_page_config(page_title="AHP Task Prioritization", layout="wide")

# ---------- Constants & Helpers ----------
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

def geometric_mean_weights(A: np.ndarray) -> np.ndarray:
    gm = np.prod(A, axis=1) ** (1.0 / A.shape[0])
    w = gm / gm.sum()
    return w

def principal_eigenvector_weights(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(np.real(vals))
    v = np.real(vecs[:, idx])
    v = np.abs(v)
    w = v / v.sum()
    return w

def lambda_max(A: np.ndarray, w: np.ndarray) -> float:
    Aw = A @ w
    ratios = Aw / (w + 1e-12)
    return float(np.mean(ratios))

def consistency_ratio(A: np.ndarray, w: np.ndarray):
    n = A.shape[0]
    lmax = lambda_max(A, w)
    CI = (lmax - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / (RI if RI != 0 else 1e-12)
    return float(CR), float(CI), float(lmax)

def saaty_label(val: int) -> str:
    mapping = {
        1: "1 - Equal importance",
        2: "2 - Between equal & moderate",
        3: "3 - Moderate importance",
        4: "4 - Between moderate & strong",
        5: "5 - Strong importance",
        6: "6 - Between strong & very strong",
        7: "7 - Very strong importance",
        8: "8 - Between very strong & extreme",
        9: "9 - Extreme importance",
    }
    return mapping.get(val, str(val))

st.title("AHP Task Prioritization")

# ---------- 1) Criteria ----------
with st.expander("1) Define Criteria", expanded=True):
    default_criteria = ["Urgency", "Complexity", "Impact", "Stakeholder Importance"]
    criteria = st_tags(
        label="Criteria (press Enter to add)",
        text="Press Enter to add more",
        value=default_criteria,
        suggestions=[],
        maxtags=10,
        key="criteria_tags",
    )
    criteria = [c for c in criteria if str(c).strip()]
    if len(criteria) < 2:
        st.warning("Please keep at least two criteria.")
    method = st.selectbox("Weighting method", ["Geometric Mean (robust)", "Principal Eigenvector (classic AHP)"])
    st.caption("Tip: Keep criteria independent. Mark cost criteria in step 3 (e.g., Complexity).")

# ---------- 2) Pairwise Comparison ----------
with st.expander("2) Pairwise Comparison (Saaty 1–9 scale)", expanded=True):
    n = len(criteria)
    if n >= 2:
        A = np.ones((n, n), dtype=float)

        # Presets to match the screenshot defaults
        preset_pairs = {
            ("Urgency", "Complexity"): ("Urgency", 2),
            ("Urgency", "Impact"): ("Impact", 4),
            ("Urgency", "Stakeholder Importance"): ("Stakeholder Importance", 2),
            ("Complexity", "Impact"): ("Impact", 8),
            ("Complexity", "Stakeholder Importance"): ("Stakeholder Importance", 4),
            ("Impact", "Stakeholder Importance"): ("Impact", 4),
        }
        use_presets = criteria == ["Urgency", "Complexity", "Impact", "Stakeholder Importance"]

        cols = st.columns(min(4, n))
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for idx, (i, j) in enumerate(pairs):
            with cols[idx % len(cols)]:
                ci, cj = criteria[i], criteria[j]
                st.markdown(f"**{ci} vs {cj}**")

                # Default selection & magnitude
                if use_presets and (ci, cj) in preset_pairs:
                    preset_dir, preset_mag = preset_pairs[(ci, cj)]
                    default_index = 0 if preset_dir == ci else 1
                    default_mag = int(preset_mag)
                else:
                    default_index = 0   # choose left by default
                    default_mag = 3     # moderate importance

                direction = st.radio(
                    "Which is more important?",
                    (ci, cj),
                    key=f"dir_{i}_{j}",
                    horizontal=True,
                    index=default_index,
                )
                mag = st.slider(
                    "How much more important? (1–9)",
                    1, 9, default_mag,
                    key=f"mag_{i}_{j}",
                )
                st.caption(saaty_label(mag))

                if direction == ci:
                    A[i, j] = float(mag)
                    A[j, i] = 1.0 / float(mag)
                else:
                    A[i, j] = 1.0 / float(mag)
                    A[j, i] = float(mag)

        # Weights
        if method.startswith("Geometric"):
            w = geometric_mean_weights(A)
        else:
            try:
                w = principal_eigenvector_weights(A)
            except Exception as e:
                st.error(f"Eigenvector method failed: {e}. Falling back to Geometric Mean.")
                w = geometric_mean_weights(A)

        # Consistency
        CR, CI, LMAX = consistency_ratio(A, w)

        weights_df = pd.DataFrame({
            "Criterion": criteria,
            "Weight": np.round(w, 4)
        }).sort_values("Weight", ascending=False).reset_index(drop=True)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Weights")
            st.dataframe(weights_df, use_container_width=True)
        with right:
            st.subheader("Consistency")
            st.metric("λmax", f"{LMAX:.4f}")
            st.metric("CI", f"{CI:.4f}")
            ok = CR < 0.10
            st.metric("CR", f"{CR:.4f}", delta=("OK" if ok else "High"))
            if not ok:
                st.info("Consider revising comparisons to reduce inconsistency (CR < 0.10 preferred).")

        st.download_button(
            "Download Weights (CSV)",
            data=weights_df.to_csv(index=False),
            file_name="ahp_criteria_weights.csv",
            mime="text/csv",
        )

# ---------- 3) Task Scoring & Ranking ----------
with st.expander("3) Task Scoring & Ranking", expanded=True):
    if len(criteria) >= 2:
        st.caption("Enter task scores per criterion (1–9). For cost criteria, values are inverted as (10 - score).")
        cost_criteria = st.multiselect("Select cost criteria (higher is worse)", options=criteria, default=[c for c in criteria if c.lower().startswith("complex")])
        st.write("**Cost criteria:**", ", ".join(cost_criteria) if cost_criteria else "None")

        # Sample tasks
        sample_rows = [
            {"Task Name": "Fix ETL Failure"},
            {"Task Name": "AB Test Analysis"},
            {"Task Name": "Build Marketing Dashboard"},
            {"Task Name": "Data Quality Audit"},
        ]
        for r in sample_rows:
            for c in criteria:
                r[c] = int(np.random.randint(1, 6))

        start_mode = st.radio("Start with sample tasks?", ["Yes (sample)", "No (empty)"], horizontal=True)
        if start_mode.startswith("Yes"):
            tasks_df = pd.DataFrame(sample_rows)
        else:
            tasks_df = pd.DataFrame(columns=["Task Name"] + criteria)

        st.write("Edit tasks below:")
        edited_df = st.data_editor(
            tasks_df,
            num_rows="dynamic",
            use_container_width=True,
            key="task_editor",
        )

        if len(edited_df) > 0:
            # Sanitize inputs
            for c in criteria:
                edited_df[c] = pd.to_numeric(edited_df[c], errors="coerce")
            edited_df = edited_df.fillna(0)

            # Adjust for cost
            adjusted = edited_df.copy()
            for c in criteria:
                if c in cost_criteria:
                    adjusted[c] = 10 - adjusted[c]

            # Build weight map
            if 'w' in locals():
                weight_map = {crit: float(w[i]) for i, crit in enumerate(criteria)}
            else:
                weight_map = {crit: 1.0/len(criteria) for crit in criteria}
            weights_vec = np.array([weight_map[c] for c in criteria])

            # Scores & ranks
            adjusted_vals = adjusted[criteria].to_numpy(dtype=float)
            scores = adjusted_vals @ weights_vec

            result = edited_df.copy()
            result["Final Score"] = np.round(scores, 4)
            result["Rank"] = result["Final Score"].rank(ascending=False, method="min").astype(int)
            result = result.sort_values(["Rank", "Final Score"], ascending=[True, False]).reset_index(drop=True)

            st.subheader("Ranking")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download Ranking (CSV)",
                data=result.to_csv(index=False),
                file_name="ahp_task_ranking.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Tip: Keep CR < 0.10 for reliable weights. Set cost criteria carefully (e.g., Complexity).")
