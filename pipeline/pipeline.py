"""The pipeline definition: ingest -> train -> evaluate -> gate -> register.

Compile it (no GCP account needed — compilation is pure IR generation):

    python -m pipeline.pipeline        # writes pipeline.yaml in the repo root
"""

from kfp import compiler, dsl

from pipeline.components import (
    evaluate,
    ingest,
    metric_gate,
    register_model,
    train,
)


@dsl.pipeline(
    name="customer-churn-training",
    description=(
        "Trains a churn classifier, gates deployment on held-out accuracy, "
        "and registers the model only when the gate passes."
    ),
)
def churn_pipeline(
    csv_uri: str = "data/customers.csv",
    accuracy_threshold: float = 0.75,
    test_size: float = 0.25,
    random_state: int = 42,
):
    ingest_task = ingest(csv_uri=csv_uri)

    train_task = train(
        dataset=ingest_task.outputs["dataset"],
        test_size=test_size,
        random_state=random_state,
    )

    evaluate_task = evaluate(
        dataset=ingest_task.outputs["dataset"],
        model=train_task.outputs["model"],
        test_size=test_size,
        random_state=random_state,
    )

    gate_task = metric_gate(
        accuracy=evaluate_task.outputs["Output"],
        threshold=accuracy_threshold,
    )

    # The deploy step exists in the graph but executes only on gate PASS —
    # a model that misses the accuracy bar is a training result, not a release.
    with dsl.If(gate_task.outputs["Output"] == True, name="accuracy-gate-passed"):
        register_model(
            model=train_task.outputs["model"],
            accuracy=evaluate_task.outputs["Output"],
        )


def compile_pipeline(output_path: str = "pipeline.yaml") -> str:
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path=output_path,
    )
    return output_path


if __name__ == "__main__":
    path = compile_pipeline()
    print(f"compiled -> {path}")
