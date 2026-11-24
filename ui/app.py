"""Streamlit UI for the DR Digital Twin.

Loads trained models and real OLIVES longitudinal sequences.
Run the full pipeline first: python run_pipeline.py

Usage: streamlit run ui/app.py
"""
import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import BIOMARKERS_6, TREATMENT_TYPES, N_BM, OUTPUT_DIR, CHECKPOINT_DIR
from digital_twin.twin import PatientDigitalTwin
from layer1_biomarker.graph import build_graph_bundle
from layer2_progression.causal_graph import build_full_causal_analysis


st.set_page_config(
    page_title="DR Digital Twin — Virtual Drug Testing",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_twin():
    """Load digital twin from trained checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NPMI matrix
    npmi_path = os.path.join(OUTPUT_DIR, "npmi_matrix.npy")
    if not os.path.exists(npmi_path):
        st.error(
            "No trained models found. Run the full pipeline first:\n\n"
            "```\npython run_pipeline.py\n```"
        )
        st.stop()

    npmi = np.load(npmi_path)
    graph_bundle = build_graph_bundle(npmi)

    # Load progression model
    prog_ckpt = os.path.join(CHECKPOINT_DIR, "progression_graph_lstm.pt")
    from layer2_progression.models import build_progression_model
    progression_model = build_progression_model("graph_lstm", npmi_matrix=npmi)
    progression_model.load_state_dict(torch.load(prog_ckpt, map_location=device))

    # Load biomarker model if available
    biomarker_model = None
    biomarker_thresholds = None
    bm_ckpt = os.path.join(CHECKPOINT_DIR, "best_biomarker_model.pt")
    if os.path.exists(bm_ckpt):
        from layer1_biomarker.models import build_model
        import json
        results_path = os.path.join(OUTPUT_DIR, "metrics", "layer1_results.json")
        best_model_name = "gcn"
        if os.path.exists(results_path):
            results = json.load(open(results_path))
            best = max(results, key=lambda r: r["macro_f1"])
            best_model_name = best["model"]
        biomarker_model = build_model(best_model_name, graph_bundle)
        biomarker_model.load_state_dict(torch.load(bm_ckpt, map_location=device))

        thresh_path = os.path.join(CHECKPOINT_DIR, "biomarker_thresholds.npy")
        if os.path.exists(thresh_path):
            biomarker_thresholds = np.load(thresh_path)

    twin = PatientDigitalTwin(
        biomarker_model=biomarker_model,
        progression_model=progression_model,
        graph_bundle=graph_bundle,
        npmi_matrix=npmi,
        device=device,
        biomarker_thresholds=biomarker_thresholds,
    )
    return twin


@st.cache_data
def load_real_sequences():
    """Load real longitudinal sequences from the pipeline output."""
    seq_path = os.path.join(OUTPUT_DIR, "longitudinal_sequences.pkl")
    if not os.path.exists(seq_path):
        st.error(
            "No longitudinal sequences found. Run the full pipeline first:\n\n"
            "```\npython run_pipeline.py\n```"
        )
        return None
    with open(seq_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def compute_causal_analysis(_sequences):
    """Run causal analysis on real sequences (cached)."""
    return build_full_causal_analysis(_sequences, BIOMARKERS_6, N_BM)


def render_biomarker_radar(biomarker_probs, biomarker_names):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=biomarker_probs + [biomarker_probs[0]],
        theta=biomarker_names + [biomarker_names[0]],
        fill='toself',
        name='Biomarker Probability',
        line=dict(color='#e74c3c', width=2),
        fillcolor='rgba(231, 76, 60, 0.2)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def render_progression_chart(results, biomarker_names):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Biomarker Trajectories", "Severity Score"])

    colors = px.colors.qualitative.Set2
    probs = results["probabilities"]
    steps = list(range(len(probs)))

    for i, bm in enumerate(biomarker_names):
        fig.add_trace(go.Scatter(
            x=steps, y=probs[:, i],
            mode='lines+markers', name=bm,
            line=dict(color=colors[i % len(colors)], width=2),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=results["severity_score"],
        mode='lines+markers', name='Severity',
        line=dict(color='#e74c3c', width=3),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)',
    ), row=1, col=2)

    fig.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(title_text="Visit", row=1, col=1)
    fig.update_xaxes(title_text="Visit", row=1, col=2)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1], row=1, col=2)
    return fig


def render_treatment_comparison(comparison_results, biomarker_names):
    treatments = [t for t in comparison_results if not t.startswith("_")]
    summary = comparison_results["_summary"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Final Severity by Treatment", "Severity Trajectories"],
    )

    severities = [summary["final_severity"][t] for t in treatments]
    colors = ['#2ecc71' if t == summary["recommendation"] else '#95a5a6' for t in treatments]
    fig.add_trace(go.Bar(
        x=treatments, y=severities,
        marker_color=colors, name='Final Severity',
        text=[f"{s:.3f}" for s in severities], textposition='auto',
    ), row=1, col=1)

    line_colors = px.colors.qualitative.Set1
    for i, treatment in enumerate(treatments):
        r = comparison_results[treatment]
        fig.add_trace(go.Scatter(
            x=list(range(len(r["severity_score"]))),
            y=r["severity_score"],
            mode='lines+markers', name=treatment,
            line=dict(color=line_colors[i % len(line_colors)], width=2,
                      dash='dash' if treatment == 'observation' else 'solid'),
        ), row=1, col=2)

    fig.update_layout(height=400, showlegend=True, margin=dict(l=40, r=40, t=40, b=40))
    fig.update_yaxes(title_text="Severity Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Severity Score", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="Visit", row=1, col=2)
    return fig


def render_effectiveness_table(effectiveness):
    rows = []
    for e in effectiveness:
        rows.append({
            "Treatment": e["treatment"],
            "Initial Active": e["initial_active_biomarkers"],
            "Final Active": e["final_active_biomarkers"],
            "Severity Change": f"{e['severity_change']:+.3f}",
            "Recommended": "Yes" if e["is_recommended"] else "",
        })
    return rows


def main():
    st.title("Digital Twin for Diabetic Retinopathy")
    st.markdown("**Graph-Aware Biomarker Detection, Progression Modeling & Virtual Drug Testing**")
    st.caption("All results computed from real OLIVES dataset — no simulated data")

    twin = load_twin()
    seq_data = load_real_sequences()

    st.sidebar.header("Patient Configuration")

    # If we have real patient data, let user pick a real patient
    if seq_data is not None:
        all_seqs = seq_data["all"]
        patient_ids = [s["patient_id"] for s in all_seqs]
        unique_patients = sorted(set(patient_ids), key=str)

        use_real = st.sidebar.checkbox("Use real patient from OLIVES", value=True)

        if use_real and unique_patients:
            selected_patient = st.sidebar.selectbox("Select patient", unique_patients)
            patient_seq = [s for s in all_seqs if s["patient_id"] == selected_patient][0]
            twin.set_patient(str(selected_patient))

            # Use REAL biomarker values from the selected patient's LAST visit
            bm_array = patient_seq["biomarker_seq"][-1].astype(np.float32)
            clinical = np.nan_to_num(patient_seq["clinical_seq"][-1], nan=0.0).astype(np.float32)
            biomarker_values = {bm: float(bm_array[i]) for i, bm in enumerate(BIOMARKERS_6)}

            st.sidebar.subheader("Real Biomarker State (Last Visit)")
            for bm in BIOMARKERS_6:
                val = biomarker_values[bm]
                st.sidebar.write(f"{'🔴' if val > 0.5 else '🟢'} **{bm}**: {'Active' if val > 0.5 else 'Inactive'}")

            n_visits = patient_seq["biomarker_seq"].shape[0]
            st.sidebar.metric("Total Visits", n_visits)
        else:
            use_real = False
    else:
        use_real = False

    if not use_real:
        patient_id = st.sidebar.text_input("Patient ID", value="MANUAL-001")
        twin.set_patient(patient_id)

        st.sidebar.subheader("Biomarker State")
        biomarker_values = {}
        for bm in BIOMARKERS_6:
            biomarker_values[bm] = st.sidebar.slider(bm, 0.0, 1.0, 0.5, 0.05, key=f"bm_{bm}")

        bm_array = np.array([biomarker_values[bm] for bm in BIOMARKERS_6], dtype=np.float32)
        bm_array = (bm_array > 0.5).astype(np.float32)

        st.sidebar.subheader("Clinical Features")
        bcva = st.sidebar.number_input("BCVA (letters)", value=60, min_value=0, max_value=100)
        cst = st.sidebar.number_input("CST (μm)", value=300, min_value=100, max_value=800)
        clinical = np.array([bcva, cst], dtype=np.float32)

    twin.add_visit(biomarkers=bm_array, clinical_data=clinical)

    # ── Main Content ──────────────────────────────────────────────────

    tab1, tab2, tab3, tab5, tab4 = st.tabs([
        "Current State", "Disease Progression", "Treatment Comparison",
        "Causal Graph", "Patient Report",
    ])

    with tab1:
        st.subheader("Biomarker Profile")
        col1, col2 = st.columns([1, 1])

        with col1:
            fig = render_biomarker_radar(
                [biomarker_values[bm] for bm in BIOMARKERS_6],
                list(BIOMARKERS_6),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            active = [bm for bm in BIOMARKERS_6 if biomarker_values[bm] > 0.5]
            severity = np.mean(list(biomarker_values.values()))

            st.metric("Overall Severity", f"{severity:.2f}", delta=None)
            st.metric("Active Biomarkers", f"{len(active)} / {N_BM}")

            if active:
                st.write("**Active:**", ", ".join(active))
            else:
                st.success("No active biomarkers detected")

            if biomarker_values.get("IRF", 0) > 0.5 and biomarker_values.get("DRT/ME", 0) > 0.5:
                st.info("IRF and DRT/ME are co-occurring (NPMI +0.661) — consistent with active DME")
            if biomarker_values.get("PAVF", 0) > 0.5 and biomarker_values.get("FAVF", 0) > 0.5:
                st.warning("PAVF and FAVF are both active — these are typically mutually exclusive (NPMI -0.774)")

    with tab2:
        st.subheader("Disease Progression Prediction")

        col1, col2 = st.columns([2, 1])
        with col2:
            treatment = st.selectbox("Treatment scenario", TREATMENT_TYPES)
            n_visits = st.slider("Prediction horizon (visits)", 2, 12, 6)

        progression = twin.predict_progression(n_visits, treatment)
        fig = render_progression_chart(progression, BIOMARKERS_6)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Progression Data"):
            import pandas as pd
            df = pd.DataFrame(
                progression["probabilities"],
                columns=BIOMARKERS_6,
                index=[f"Visit {i}" for i in range(len(progression["probabilities"]))],
            )
            st.dataframe(df.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1))

    with tab3:
        st.subheader("Virtual Drug Testing — Treatment Comparison")

        n_visits_cmp = st.slider("Simulation horizon", 2, 12, 6, key="cmp_visits")

        comparison = twin.compare_treatments(n_future_visits=n_visits_cmp)
        recommendation = comparison["_summary"]["recommendation"]

        st.success(f"**Recommended treatment: {recommendation}** "
                   f"(lowest predicted severity at visit {n_visits_cmp})")

        fig = render_treatment_comparison(comparison, BIOMARKERS_6)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Effectiveness Summary")
        from layer3_treatment.simulator import compute_treatment_effectiveness
        effectiveness = comparison["_effectiveness"]
        rows = render_effectiveness_table(effectiveness)
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Temporal Causal Biomarker Graph")
        st.caption("Directed causal transitions computed from **real OLIVES longitudinal data**")

        if seq_data is None:
            st.error("No longitudinal sequences found. Run the full pipeline first.")
        else:
            all_seqs = seq_data["all"]
            st.info(f"Causal analysis based on **{len(all_seqs)} real patient-eye sequences** from OLIVES")

            causal = compute_causal_analysis(all_seqs)

            col_c1, col_c2 = st.columns([1, 1])

            with col_c1:
                ce = causal["causal_effect_matrix"]
                fig_heat = go.Figure(data=go.Heatmap(
                    z=ce, x=BIOMARKERS_6, y=BIOMARKERS_6,
                    colorscale="RdBu_r", zmid=0,
                    text=np.round(ce, 3).astype(str), texttemplate="%{text}",
                    colorbar=dict(title="Causal<br>Effect"),
                ))
                fig_heat.update_layout(
                    title="Causal Effect Matrix (row → col)",
                    xaxis_title="Target biomarker (t+1)",
                    yaxis_title="Source biomarker (t)",
                    height=400, margin=dict(l=60, r=20, t=40, b=60),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            with col_c2:
                edges = causal["significant_edges"]
                if edges:
                    import networkx as nx
                    G = nx.DiGraph()
                    for bm in BIOMARKERS_6:
                        G.add_node(bm)
                    for e in edges:
                        G.add_edge(e["source"], e["target"], weight=abs(e["causal_effect"]))

                    pos = nx.circular_layout(G)
                    annotations = []
                    for src, dst, d in G.edges(data=True):
                        x0, y0 = pos[src]
                        x1, y1 = pos[dst]
                        annotations.append(dict(
                            ax=x0, ay=y0, x=x1, y=y1,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5,
                            arrowwidth=max(1, d["weight"] * 8),
                            arrowcolor="rgba(231,76,60,0.6)",
                        ))

                    node_x = [pos[n][0] for n in G.nodes()]
                    node_y = [pos[n][1] for n in G.nodes()]

                    fig_graph = go.Figure()
                    fig_graph.add_trace(go.Scatter(
                        x=node_x, y=node_y, mode='markers+text',
                        text=list(G.nodes()), textposition="top center",
                        marker=dict(size=20, color='#3498db', line=dict(width=2, color='white')),
                    ))
                    fig_graph.update_layout(
                        title=f"Significant Causal Edges ({len(edges)} links, p<0.05)",
                        showlegend=False, height=400,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=annotations,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_graph, use_container_width=True)
                else:
                    st.info("No statistically significant causal edges found (p<0.05).")

            # Treatment-conditioned transitions
            if causal["treatment_transitions"]:
                st.subheader("Treatment-Conditioned Causal Effects")
                st.caption("How each treatment alters the biomarker causal structure (from real data)")

                treat_tabs = st.tabs(list(causal["treatment_transitions"].keys()))
                for tab_t, (treat_label, treat_data) in zip(treat_tabs, causal["treatment_transitions"].items()):
                    with tab_t:
                        ce_t = treat_data["causal_effect"]
                        fig_t = go.Figure(data=go.Heatmap(
                            z=ce_t, x=BIOMARKERS_6, y=BIOMARKERS_6,
                            colorscale="RdBu_r", zmid=0,
                            text=np.round(ce_t, 3).astype(str), texttemplate="%{text}",
                        ))
                        fig_t.update_layout(
                            title=f"Treatment: {treat_label} (n={treat_data['n_sequences']} patients)",
                            height=350, margin=dict(l=60, r=20, t=40, b=60),
                        )
                        st.plotly_chart(fig_t, use_container_width=True)

            # Causal chains
            if causal["causal_chains"]:
                st.subheader("Discovered Causal Chains")
                import pandas as pd
                chain_rows = []
                for c in causal["causal_chains"][:10]:
                    chain_rows.append({
                        "Chain": " → ".join(c["chain"]),
                        "Strength": f"{c['total_strength']:.3f}",
                        "Length": c["length"],
                    })
                st.dataframe(pd.DataFrame(chain_rows), use_container_width=True, hide_index=True)

            # Interventional analysis
            st.subheader("Interventional Analysis")
            st.caption("What happens downstream if we block a specific biomarker?")
            intervene_bm = st.selectbox("Block biomarker:", BIOMARKERS_6)
            intervention = causal["interventions"].get(intervene_bm, {})
            if intervention:
                total_fx = intervention.get("total_effects", {})
                import pandas as pd
                int_rows = [{"Downstream Biomarker": k, "Expected Change": f"{v:+.3f}"} for k, v in total_fx.items()]
                st.dataframe(pd.DataFrame(int_rows), use_container_width=True, hide_index=True)
                st.info(f"Most affected: **{intervention.get('most_affected', 'N/A')}**")

    with tab4:
        st.subheader("Patient Digital Twin Report")

        summary = twin.get_patient_summary()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient ID", summary["patient_id"])
        with col2:
            st.metric("Total Visits", summary["n_visits"])
        with col3:
            st.metric("Current Severity", f"{summary['current_state']['severity_score']:.3f}")

        st.json(summary)

        if st.button("Export Full Report (JSON)"):
            path = twin.export_report()
            st.success(f"Report exported to: {path}")


if __name__ == "__main__":
    main()
