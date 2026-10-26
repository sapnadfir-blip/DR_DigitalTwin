"""Temporal Causal Biomarker Graph — the core novelty.

Instead of static NPMI co-occurrence ("IRF and DRT/ME appear together"),
this builds DIRECTED temporal transition graphs from longitudinal visits:

    "IRF at visit t → DRT/ME at visit t+1 with probability P"

Key innovations:
1. Treatment-conditioned transition matrices (how does anti-VEGF change the causal chain?)
2. Granger-style causal testing between biomarker pairs
3. Causal chain discovery (which biomarkers drive disease progression?)
4. Interventional reasoning (if we block biomarker X, what downstream effect?)
"""
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict


def compute_temporal_transitions(sequences, n_biomarkers=6):
    """Compute biomarker transition probabilities from longitudinal visits.

    For each pair (i, j), compute:
        P(biomarker_j = 1 at t+1 | biomarker_i = 1 at t)
        P(biomarker_j = 1 at t+1 | biomarker_i = 0 at t)

    The causal signal is the DIFFERENCE: if biomarker_i being active at t
    raises the probability of biomarker_j at t+1, that's a causal lead.

    Returns:
        trans_matrix: (n_bm, n_bm) — P(j=1 at t+1 | i=1 at t)
        baseline_matrix: (n_bm, n_bm) — P(j=1 at t+1 | i=0 at t)
        causal_effect: (n_bm, n_bm) — trans_matrix - baseline_matrix (the lift)
        counts: (n_bm, n_bm, 2) — sample counts for [i=0, i=1] conditioning
    """
    # Accumulators: count[i][j][condition_i] = (n_j_active, n_total)
    active_counts = np.zeros((n_biomarkers, n_biomarkers, 2))   # j=1 counts
    total_counts = np.zeros((n_biomarkers, n_biomarkers, 2))    # total counts

    for seq in sequences:
        bm = seq["biomarker_seq"] if isinstance(seq, dict) else seq
        T = bm.shape[0]
        for t in range(T - 1):
            state_t = bm[t]
            state_t1 = bm[t + 1]
            for i in range(n_biomarkers):
                cond = int(state_t[i])  # 0 or 1
                for j in range(n_biomarkers):
                    total_counts[i, j, cond] += 1
                    if state_t1[j] > 0.5:
                        active_counts[i, j, cond] += 1

    # Transition probabilities
    trans_matrix = np.zeros((n_biomarkers, n_biomarkers))     # P(j|i=1)
    baseline_matrix = np.zeros((n_biomarkers, n_biomarkers))  # P(j|i=0)

    for i in range(n_biomarkers):
        for j in range(n_biomarkers):
            if total_counts[i, j, 1] > 0:
                trans_matrix[i, j] = active_counts[i, j, 1] / total_counts[i, j, 1]
            if total_counts[i, j, 0] > 0:
                baseline_matrix[i, j] = active_counts[i, j, 0] / total_counts[i, j, 0]

    causal_effect = trans_matrix - baseline_matrix

    return trans_matrix, baseline_matrix, causal_effect, np.stack([total_counts, active_counts])


def compute_treatment_conditioned_transitions(sequences, n_biomarkers=6):
    """Compute separate transition matrices per treatment type.

    This reveals HOW treatments alter the causal structure:
    - Under observation: IRF(t) → DRT/ME(t+1) with P=0.7
    - Under anti-VEGF:  IRF(t) → DRT/ME(t+1) with P=0.2  (treatment blocks it)

    Returns:
        dict mapping treatment_label → (trans_matrix, baseline_matrix, causal_effect)
    """
    # Group sequences by treatment
    treatment_groups = defaultdict(list)
    for seq in sequences:
        if isinstance(seq, dict):
            treat = seq.get("treatment_seq")
            if treat is not None:
                # Use the most common treatment in the sequence
                if hasattr(treat, '__len__'):
                    vals, counts = np.unique(treat, return_counts=True)
                    label = str(vals[np.argmax(counts)])
                else:
                    label = str(treat)
            else:
                label = "unknown"
            treatment_groups[label].append(seq)
        else:
            treatment_groups["unknown"].append({"biomarker_seq": seq})

    results = {}
    for label, group in treatment_groups.items():
        if len(group) < 3:
            continue
        trans, baseline, causal, _ = compute_temporal_transitions(group, n_biomarkers)
        results[label] = {
            "transition_matrix": trans,
            "baseline_matrix": baseline,
            "causal_effect": causal,
            "n_sequences": len(group),
        }

    return results


def granger_causal_test(sequences, n_biomarkers=6, max_lag=2):
    """Granger-style causality test between biomarker pairs.

    Tests whether knowing biomarker_i at time (t - lag) improves prediction
    of biomarker_j at time t, beyond what biomarker_j's own history provides.

    Uses chi-squared test on contingency tables (since biomarkers are binary).

    Returns:
        granger_matrix: (n_bm, n_bm) — p-values (low = significant causal link)
        effect_sizes: (n_bm, n_bm) — Cramér's V effect size
    """
    granger_pvals = np.ones((n_biomarkers, n_biomarkers))
    effect_sizes = np.zeros((n_biomarkers, n_biomarkers))

    for i in range(n_biomarkers):
        for j in range(n_biomarkers):
            if i == j:
                continue

            # Build contingency: [i_state_at_t-1] × [j_state_at_t]
            # conditioned on j_state_at_t-1 (to control for j's own history)
            contingency = np.zeros((2, 2, 2))  # [j_prev][i_prev][j_next]

            for seq in sequences:
                bm = seq["biomarker_seq"] if isinstance(seq, dict) else seq
                T = bm.shape[0]
                for lag in range(1, min(max_lag + 1, T)):
                    for t in range(lag, T):
                        j_prev = int(bm[t - lag, j])
                        i_prev = int(bm[t - lag, i])
                        j_next = int(bm[t, j])
                        contingency[j_prev, i_prev, j_next] += 1

            # Test for each j_prev stratum, then combine (Cochran-Mantel-Haenszel style)
            best_pval = 1.0
            best_v = 0.0
            for j_prev in range(2):
                table = contingency[j_prev]  # (2, 2): [i_prev] × [j_next]
                if table.sum() < 10:
                    continue
                try:
                    chi2, pval, _, _ = stats.chi2_contingency(table, correction=True)
                    n = table.sum()
                    v = np.sqrt(chi2 / max(n, 1))  # Cramér's V
                    if pval < best_pval:
                        best_pval = pval
                        best_v = v
                except ValueError:
                    continue

            granger_pvals[i, j] = best_pval
            effect_sizes[i, j] = best_v

    return granger_pvals, effect_sizes


def discover_causal_chains(causal_effect, biomarker_names, threshold=0.1):
    """Find directed causal chains in the biomarker transition graph.

    A chain A → B → C means:
    - A being active at t raises P(B active at t+1) by > threshold
    - B being active at t raises P(C active at t+1) by > threshold

    Returns list of chains sorted by total causal strength.
    """
    n = causal_effect.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and causal_effect[i, j] > threshold:
                edges.append((i, j, causal_effect[i, j]))

    # Build adjacency list
    adj = defaultdict(list)
    for src, dst, weight in edges:
        adj[src].append((dst, weight))

    # Find chains of length 2 and 3 via DFS
    chains = []
    for start in range(n):
        for mid, w1 in adj[start]:
            chains.append({
                "chain": [biomarker_names[start], biomarker_names[mid]],
                "effects": [w1],
                "total_strength": w1,
                "length": 2,
            })
            for end, w2 in adj[mid]:
                if end != start:
                    chains.append({
                        "chain": [biomarker_names[start], biomarker_names[mid], biomarker_names[end]],
                        "effects": [w1, w2],
                        "total_strength": w1 + w2,
                        "length": 3,
                    })

    chains.sort(key=lambda c: c["total_strength"], reverse=True)
    return chains


def interventional_analysis(causal_effect, biomarker_names, intervene_on, block=True):
    """Estimate the downstream effect of intervening on a biomarker.

    If we BLOCK biomarker X (e.g., anti-VEGF resolves IRF), what is the
    expected reduction in downstream biomarkers?

    This is the causal version of "what if we treat the root cause?"

    Args:
        causal_effect: (n_bm, n_bm) directed causal effect matrix
        intervene_on: index or name of the biomarker to intervene on
        block: True = remove the biomarker, False = activate it

    Returns:
        dict of downstream effects per biomarker
    """
    if isinstance(intervene_on, str):
        idx = biomarker_names.index(intervene_on)
    else:
        idx = intervene_on

    n = causal_effect.shape[0]

    # Direct effects
    direct = {}
    for j in range(n):
        if j == idx:
            continue
        effect = causal_effect[idx, j]
        if block:
            effect = -effect  # blocking removes the causal influence
        direct[biomarker_names[j]] = float(effect)

    # Indirect effects (2-hop): intervene_on → intermediate → target
    indirect = {}
    for j in range(n):
        if j == idx:
            continue
        total_indirect = 0.0
        for k in range(n):
            if k == idx or k == j:
                continue
            path_effect = causal_effect[idx, k] * causal_effect[k, j]
            if block:
                path_effect = -path_effect
            total_indirect += path_effect
        indirect[biomarker_names[j]] = float(total_indirect)

    # Total effect = direct + indirect
    total = {}
    for name in direct:
        total[name] = direct[name] + indirect.get(name, 0.0)

    return {
        "intervention": f"{'Block' if block else 'Activate'} {biomarker_names[idx]}",
        "direct_effects": direct,
        "indirect_effects": indirect,
        "total_effects": total,
        "most_affected": max(total, key=lambda k: abs(total[k])),
    }


def build_full_causal_analysis(sequences, biomarker_names, n_biomarkers=6):
    """Run the complete causal analysis pipeline.

    Returns a comprehensive causal analysis report.
    """
    # 1. Basic temporal transitions
    trans, baseline, causal_effect, counts = compute_temporal_transitions(
        sequences, n_biomarkers
    )

    # 2. Treatment-conditioned transitions
    treatment_transitions = compute_treatment_conditioned_transitions(
        sequences, n_biomarkers
    )

    # 3. Granger causality tests
    granger_pvals, granger_effects = granger_causal_test(
        sequences, n_biomarkers
    )

    # 4. Causal chain discovery
    chains = discover_causal_chains(causal_effect, biomarker_names)

    # 5. Interventional analysis for each biomarker
    interventions = {}
    for i, name in enumerate(biomarker_names):
        interventions[name] = interventional_analysis(
            causal_effect, biomarker_names, i, block=True
        )

    # 6. Summary statistics
    significant_edges = []
    for i in range(n_biomarkers):
        for j in range(n_biomarkers):
            if i != j and granger_pvals[i, j] < 0.05 and abs(causal_effect[i, j]) > 0.05:
                significant_edges.append({
                    "source": biomarker_names[i],
                    "target": biomarker_names[j],
                    "causal_effect": float(causal_effect[i, j]),
                    "p_value": float(granger_pvals[i, j]),
                    "effect_size": float(granger_effects[i, j]),
                    "direction": "promotes" if causal_effect[i, j] > 0 else "inhibits",
                })
    significant_edges.sort(key=lambda e: abs(e["causal_effect"]), reverse=True)

    return {
        "transition_matrix": trans,
        "baseline_matrix": baseline,
        "causal_effect_matrix": causal_effect,
        "granger_pvalues": granger_pvals,
        "granger_effect_sizes": granger_effects,
        "treatment_transitions": treatment_transitions,
        "causal_chains": chains[:20],
        "interventions": interventions,
        "significant_edges": significant_edges,
        "biomarker_names": biomarker_names,
        "n_sequences": len(sequences),
        "n_transitions": int(counts[0].sum() / 2),
    }
