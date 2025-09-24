import mlflow

def register_models():
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name("abtest_movielens")

    # 최신 run 가져오기
    runs = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=5)

    # LogReg (Policy A)
    runA = runs[runs["tags.mlflow.runName"] == "PolicyA_LogReg"].iloc[0]
    modelA_uri = f"runs:/{runA.run_id}/model"
    mvA = mlflow.register_model(modelA_uri, "movielens_ctr_ab")
    client.set_registered_model_alias("movielens_ctr_ab", "PolicyA", mvA.version)

    # LightGBM (Policy B)
    runB = runs[runs["tags.mlflow.runName"] == "PolicyB_LightGBM"].iloc[0]
    modelB_uri = f"runs:/{runB.run_id}/model"
    mvB = mlflow.register_model(modelB_uri, "movielens_ctr_ab")
    client.set_registered_model_alias("movielens_ctr_ab", "PolicyB", mvB.version)

if __name__ == "__main__":
    register_models()
