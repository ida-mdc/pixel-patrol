import json
import random
from pathlib import Path
import argparse


def generate_ctc_metrics(base_tra=0.95, base_det=0.97):
    """Generates a realistic CTCMetrics block."""
    tra = max(0, min(1, base_tra + random.uniform(-0.05, 0.05)))
    det = max(0, min(1, base_det + random.uniform(-0.05, 0.05)))
    lnk = max(0, min(1, tra - random.uniform(0.005, 0.015)))

    fp_nodes = random.randint(0, 5)
    fn_nodes = random.randint(20, 50) * (1.05 - det)
    fp_edges = random.randint(50, 70) * (1.05 - tra)
    fn_edges = random.randint(70, 100) * (1.05 - tra)
    ws_edges = random.randint(40, 60) * (1.05 - tra)

    # AOGM is a weighted sum of errors
    aogm = (
            fp_nodes * 1 + fn_nodes * 10 +
            fp_edges * 1 + fn_edges * 1.5 + ws_edges * 1
    )

    return {
        "version": "0.3.1.dev44+ged80104.d20250726",
        "results": {
            "AOGM": round(aogm, 2),
            "fp_nodes": int(fp_nodes),
            "fn_nodes": int(fn_nodes),
            "ns_nodes": 0,
            "fp_edges": int(fp_edges),
            "fn_edges": int(fn_edges),
            "ws_edges": int(ws_edges),
            "TRA": tra,
            "DET": det,
            "LNK": lnk,
        },
        "matcher": {"name": "CTCMatcher", "matching type": "one-to-one"},
        "metric": {
            "name": "CTCMetrics",
            "valid_match_types": ["one-to-one", "many-to-one"],
            "v_weights": {"ns": 5, "fp": 1, "fn": 10},
            "e_weights": {"fp": 1, "fn": 1.5, "ws": 1},
            "relax_skips_gt": False,
            "relax_skips_pred": False
        }
    }


def generate_division_metrics(base_f1=0.7):
    """Generates a realistic DivisionMetrics block."""
    f1 = max(0, min(1, base_f1 + random.uniform(-0.1, 0.1)))
    precision = max(0, min(1, f1 + random.uniform(-0.05, 0.05)))
    recall = max(0, min(1, f1 + random.uniform(-0.05, 0.05)))

    total_gt_div = random.randint(85, 105)
    tp_div = int(total_gt_div * recall)
    fp_div = int((tp_div / precision) - tp_div) if precision > 0 else 0
    fn_div = total_gt_div - tp_div

    return {
        "version": "0.3.1.dev44+ged80104.d20250726",
        "results": {
            "Frame Buffer 0": {
                "Division Recall": recall,
                "Division Precision": precision,
                "Division F1": f1,
                "Mitotic Branching Correctness": max(0, f1 - 0.15),
                "Total GT Divisions": total_gt_div,
                "Total Predicted Divisions": tp_div + fp_div,
                "True Positive Divisions": tp_div,
                "False Positive Divisions": fp_div,
                "False Negative Divisions": fn_div,
                "Wrong Children Divisions": random.randint(3, 10)
            }
        },
        "matcher": {"name": "CTCMatcher", "matching type": "one-to-one"},
        "metric": {
            "name": "DivisionMetrics",
            "valid_match_types": ["one-to-one"],
            "frame_buffer": 0,
            "relax_skips_gt": False,
            "relax_skips_pred": False
        }
    }


def generate_basic_metrics(total_nodes=8600, total_edges=8500):
    """Generates a realistic BasicMetrics block."""
    return {
        "version": "0.3.1.dev44+ged80104.d20250726",
        "results": {
            "Total GT Nodes": total_nodes + random.randint(-50, 50),
            "Total Pred Nodes": total_nodes + random.randint(-50, 50),
            "True Positive Nodes": 0, "False Positive Nodes": 0, "False Negative Nodes": 0,
            "Node Recall": 0.0, "Node Precision": 0.0, "Node F1": float('nan'),
            "Total GT Edges": total_edges + random.randint(-50, 50),
            "Total Pred Edges": total_edges + random.randint(-50, 50),
            "True Positive Edges": 0, "False Positive Edges": 0, "False Negative Edges": 0,
            "Edge Recall": 0.0, "Edge Precision": 0.0, "Edge F1": float('nan'),
        },
        "matcher": {"name": "CTCMatcher", "matching type": "one-to-one"},
        "metric": {
            "name": "BasicMetrics",
            "valid_match_types": ["one-to-one"],
            "relax_skips_gt": False,
            "relax_skips_pred": False
        }
    }


def create_traccuracy_file(output_path: Path, gt_name: str, pred_name: str, seed: int):
    """Creates a single geff JSON file with multiple metric blocks."""
    random.seed(seed)

    # Introduce variation between files based on the seed
    base_tra = 0.85 + (seed * 0.02)
    base_det = 0.90 + (seed * 0.01)
    base_div_f1 = 0.65 + (seed * 0.03)

    ctc_block = generate_ctc_metrics(base_tra, base_det)
    div_block = generate_division_metrics(base_div_f1)
    basic_block = generate_basic_metrics()

    # Add gt and pred names to each block
    for block in [ctc_block, div_block, basic_block]:
        block["gt"] = gt_name
        block["pred"] = pred_name

    final_json = {"geff": [ctc_block, div_block, basic_block]}

    with open(output_path, 'w') as f:
        # Using allow_nan=False and replacing NaN with null for standard JSON
        # The json.dump default handles this, but explicit is clearer.
        json_str = json.dumps(final_json, indent=4).replace("NaN", "null")
        f.write(json_str)

    print(f"Generated test file: {output_path}")


if __name__ == "__main__":

    num_files = 30
    base_name = "gen"
    output_dir = Path(__file__).parent / "data" / "traccuracy_generated"

    output_dir.mkdir(exist_ok=True)

    for i in range(num_files):
        # Generate unique names for each run
        gt_name = f"{base_name}_GT"
        pred_name = f"{base_name}_RES_run_{i}"
        file_path = output_dir / f"traccuracy_{pred_name}.json"

        # Use index 'i' as the seed for reproducibility and variation
        create_traccuracy_file(file_path, gt_name, pred_name, seed=i)