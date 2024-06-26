import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from typing import Any, Dict


def calculate_average_score(oss_scores: np.ndarray, binary_predictions):
    n_samples = len(binary_predictions)
    model_predictions = binary_predictions.reshape(n_samples, 1) >= np.linspace(
        0, 1, 100
    ).reshape(1, 100)
    routing_percentage = model_predictions.mean(axis=0)

    chosen_scores = np.where(
        model_predictions,
        oss_scores[:, np.newaxis],
        np.ones_like(model_predictions) * 5,
    )
    average_score = chosen_scores.mean(axis=0)
    score_auc = sklearn.metrics.auc(routing_percentage, average_score)

    return average_score, routing_percentage, score_auc


def plot_quality_cost_curve(
    oss_scores: np.ndarray,
    closed_scores: np.ndarray,
    router_binary_predictions: Dict[str, Any],
    closed_model: str = "GPT-4",
    oss_model: str = "Mixtral",
):
    """Compute cost/quality curve preferebly on data sampled from a real distribution"""

    plt.clf()
    plt.axhline(
        y=closed_scores.mean(),
        linestyle="--",
        label=closed_model,
        color="blue",
    )

    plt.axhline(
        y=oss_scores.mean(),
        linestyle="--",
        label=oss_model,
        color="red",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(router_binary_predictions)))

    for i, (router_name, binary_probs) in enumerate(router_binary_predictions.items()):
        average_score, routing_percentage, score_auc = calculate_average_score(
            oss_scores, binary_probs
        )

        plt.plot(
            routing_percentage,
            average_score,
            label=f"{router_name} (AUC={score_auc:.2f})",
            color=colors[i],
        )

    plt.xlabel("Routing to OSS %")
    plt.ylabel("Average score")
    plt.title("Average score as routing to OSS model increases")
    plt.legend()
    plt.grid(True)
