
import os
import tempfile
import warnings

import pandas as pd
from minio import Minio

# Modelos de classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Separação de treino e teste
from sklearn.model_selection import train_test_split

# Métricas de avaliação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# MLflow para tracking dos experimentos
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# Configuração do MLflow
# Se estiver rodando o MLflow localmente, a URI será essa:
# http://localhost:3000
mlflow.set_tracking_uri("http://localhost:3000")

mlflow.set_experiment("youtube_trending_classification")

# Conexão com o MinIO
client = Minio(
    "localhost:9000",
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)


# Função para carregar os dados da camada Gold do MinIO
def load_gold_data():
    """
    Baixa o arquivo da camada Gold do MinIO, lê com pandas
    e retorna um DataFrame.
    """

    print("Baixando dados da camada Gold...")

    # Criando uma pasta temporária que será apagada no final
    with tempfile.TemporaryDirectory() as temp_dir:
        gold_path = os.path.join(temp_dir, "trending_gold.parquet")

        # Faz o download do arquivo do bucket "gold"
        client.fget_object("gold", "trending_gold.parquet", gold_path)

        # Lê o arquivo parquet
        df = pd.read_parquet(gold_path)

    print("Dados carregados com sucesso.")
    return df


# Função para preparar os dados para classificação (PRÉ-PROCESSAMENTO)
def prepare_data(df):
    """
    Realiza o pré-processamento dos dados para o modelo de classificação.
    """

    print("Iniciando pré-processamento...")

    # Criando a variável target "viral"
    # Se o vídeo teve mais de 1 milhão de views, consideramos viral
    df["viral"] = (df["views"] > 1_000_000).astype(int)

    # Criando algumas features simples a partir dos dados brutos

    # Tamanho do título
    df["title_length"] = df["title"].astype(str).apply(len)

    # Quantidade de tags
    # Se a coluna vier como "[none]", consideramos 0 tags
    df["num_tags"] = df["tags"].astype(str).apply(
        lambda x: 0 if x.strip().lower() == "[none]" else len(x.split("|"))
    )

    # Tamanho da descrição
    df["description_length"] = df["description"].astype(str).apply(len)

    # Evitar divisão por zero:
    # se views for 0, substituímos por 1 só para não quebrar a conta
    safe_views = df["views"].replace(0, 1)

    # Razão likes / views
    df["likes_ratio"] = df["likes"] / safe_views

    # Razão comments / views
    df["comments_ratio"] = df["comments"] / safe_views

    # Features finais para o modelo
    features = [
    "likes",
    "dislikes",
    "comments",
    "title_length",
    "num_tags",
    "description_length",
    "likes_ratio",
    "comments_ratio"
]

    available_features = [col for col in features if col in df.columns]

    print("Features usadas:", available_features)

    X = df[available_features].fillna(0)

    X = df[features].fillna(0)
    y = df["viral"]

    print("Pré-processamento concluído.")
    return X, y


# Função para treinar, avaliar e registrar o modelo no MLflow
def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Treina um modelo, faz previsões, calcula métricas e registra tudo no MLflow.

    Parâmetros:
        model_name (str): nome do modelo
        model: instância do modelo
        X_train, X_test, y_train, y_test: dados de treino e teste

    Retorno:
        results (dict): métricas calculadas
    """

    print(f"\nTreinando modelo: {model_name}")

    # Inicia uma execução no MLflow
    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Metricas de avaliação
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_name", model_name)

        # Tentando registrar os parâmetros internos do modelo
        if hasattr(model, "get_params"):
            params = model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        # Registrando as métricas no MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Mlflow tem suporte nativo para modelos do scikit-learn, então podemos registrar o modelo inteiro
        # mlflow.sklearn.log_model(model, artifact_path="model")

        # Imprime as métricas no console
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, zero_division=0))

        results = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return results

def main():
    """
    Função principal do pipeline de treino.
    """
    print("Iniciando pipeline de treinamento...")
    df = load_gold_data()
    X, y = prepare_data(df)
    
    # Divindindo em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTamanho dos conjuntos:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    # Modelos a serem testados
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Treinando e avaliando cada modelo, armazenando os resultados
    all_results = []

    for model_name, model in models.items():
        result = train_and_evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        all_results.append(result)

    # Comparando os resultados dos modelos
    results_df = pd.DataFrame(all_results)

    print("\nResumo final dos modelos:")
    print(results_df.sort_values(by="f1_score", ascending=False))

    # Escolhe o melhor modelo pelo F1-score
    best_model = results_df.sort_values(by="f1_score", ascending=False).iloc[0]

    print("\nMelhor modelo:")
    print(best_model)

    print("\nPipeline concluído com sucesso.")

if __name__ == "__main__":
    main()