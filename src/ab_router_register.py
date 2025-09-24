import mlflow

def register_router():
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name("abtest_movielens")
    runs = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=5)

    # Router run 찾기
    run_router = runs[runs["tags.mlflow.runName"] == "AB_Router_Demo"].iloc[0]
    model_uri = f"runs:/{run_router.run_id}/ab_router"
    mv = mlflow.register_model(model_uri, "movielens_ctr_router")
    client.set_registered_model_alias("movielens_ctr_router", "router", mv.version)

if __name__ == "__main__":
    register_router()
