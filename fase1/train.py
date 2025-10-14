from __future__ import annotations
import csv
import random
from typing import List, Tuple


class Perceptron:
    """
    Implementação de um modelo Perceptron — um único neurônio treinado pela Regra Delta.

    Atributos:
        learning_rate (float): Taxa de aprendizado (η) usada para atualização dos pesos.
        weights (List[float]): Lista de pesos sinápticos do neurônio.
        bias (float): Termo de bias (ou limiar).
        training_log (List[List[str]]): Registro completo do processo de treinamento
            (usado para exportação em CSV).
    """

    def __init__(self, num_inputs: int, learning_rate: float = 0.1) -> None:
        """
        Inicializa o Perceptron com pesos e bias aleatórios.

        Args:
            num_inputs (int): Número de entradas (sensores).
            learning_rate (float): Taxa de aprendizado usada na Regra Delta.
        """
        self.learning_rate: float = learning_rate
        self.weights: List[float] = [round(random.uniform(-1, 1), 2) for _ in range(num_inputs)]
        self.bias: float = round(random.uniform(-1, 1), 2)
        self.training_log: List[List[str]] = []  # Armazena dados do treinamento

    def activation(self, x: float) -> int:
        """
        Função de ativação degrau binário.

        Retorna:
            1 se x >= 0, caso contrário -1.
        """
        return 1 if x >= 0 else -1

    def predict(self, inputs: List[float]) -> int:
        """
        Calcula a saída do Perceptron para uma entrada específica.

        Args:
            inputs (List[float]): Lista com os valores dos sensores.

        Retorna:
            int: Saída binária do neurônio (1 ou -1).
        """
        total_activation = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation(total_activation)

    def train(
        self,
        training_data: List[Tuple[List[float], int]],
        validation_data: List[Tuple[List[float], int]],
        max_epochs: int = 100
    ) -> None:
        """
        Treina o Perceptron utilizando a Regra Delta.

        Args:
            training_data: Conjunto de amostras de treinamento (entradas + saída esperada).
            validation_data: Conjunto de validação usado para testar o modelo.
            max_epochs: Número máximo de épocas de treinamento.
        """
        for epoch in range(1, max_epochs + 1):
            total_error = 0
            print(f"\n=== Época {epoch} ===")

            for idx, (inputs, expected) in enumerate(training_data, start=1):
                # Calcula a saída do neurônio
                output = self.predict(inputs)

                # Calcula o erro
                error = expected - output
                total_error += abs(error)

                # Atualiza os pesos segundo a Regra Delta: Δw = η * erro * x
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]

                # Atualiza o bias: Δb = η * erro
                self.bias += self.learning_rate * error

                # Arredonda pesos e bias para 2 casas decimais (tanto para print quanto para CSV)
                rounded_weights = [round(w, 2) for w in self.weights]
                rounded_bias = round(self.bias, 2)

                # Exibe informações da amostra atual
                print(f"Amostra {idx}: Entradas={[round(v, 2) for v in inputs]}, Esperado={expected}, "
                      f"Saída={output}, Erro={error}")
                print(f"Pesos: {rounded_weights}")
                print(f"Bias: {rounded_bias}\n")

                # Armazena no log para exportação (exatamente o que foi impresso)
                self.training_log.append([
                    epoch,
                    idx,
                    *[round(v, 2) for v in inputs],
                    expected,
                    output,
                    error,
                    str(rounded_weights),
                    rounded_bias
                ])

            # Avalia o desempenho do modelo no conjunto de validação
            accuracy = self.evaluate(validation_data)
            print(f"Acurácia na validação: {accuracy:.2f}%")

            # Critério de parada: acurácia total
            if accuracy == 100.0:
                print("\nTreinamento concluído com 100% de acurácia.")
                break

        # Exibe os resultados finais
        print("\n=== Treinamento Finalizado ===")
        print(f"Pesos finais: {[round(w, 2) for w in self.weights]}")
        print(f"Bias final: {round(self.bias, 2)}")

        # Exporta o log completo do treinamento para um arquivo CSV
        self.export_training_log("treinamento_log.csv")

    def evaluate(self, dataset: List[Tuple[List[float], int]]) -> float:
        """
        Avalia a acurácia do Perceptron sobre um conjunto de dados.

        Args:
            dataset: Lista de tuplas (entradas, saída esperada).

        Retorna:
            float: Percentual de acertos (%).
        """
        correct = sum(1 for inputs, expected in dataset if self.predict(inputs) == expected)
        return (correct / len(dataset)) * 100

    def export_training_log(self, filename: str) -> None:
        """
        Exporta os dados completos do treinamento para um arquivo CSV.

        Args:
            filename (str): Nome do arquivo CSV de saída.
        """
        header = [
            "Época",
            "Amostra",
            "Sensor 1",
            "Sensor 2",
            "Sensor 3",
            "Sensor 4",
            "Sensor 5",
            "Sensor 6",
            "Esperado",
            "Saída",
            "Erro",
            "Pesos",
            "Bias"
        ]
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(header)
            writer.writerows(self.training_log)
        print(f"\n📁 Dados de treinamento exportados para '{filename}' com sucesso.")


def load_dataset(filepath: str) -> List[Tuple[List[float], int]]:
    """
    Carrega o conjunto de amostras a partir de um arquivo CSV.

    Formato esperado do CSV:
        Sensor 1;Sensor 2;Sensor 3;Sensor 4;Sensor 5;Sensor 6;Controle Acionador
    """
    dataset: List[Tuple[List[float], int]] = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)  # Ignora o cabeçalho
        for row in reader:
            inputs = [float(x) for x in row[:-1]]
            target = int(row[-1])
            dataset.append((inputs, target))
    return dataset


def split_dataset(dataset: List[Tuple[List[float], int]], train_ratio: float = 0.7) -> Tuple[
    List[Tuple[List[float], int]],
    List[Tuple[List[float], int]]
]:
    """
    Embaralha e divide o conjunto de dados em subconjuntos de treino e validação.
    """
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_ratio)
    return dataset[:split_index], dataset[split_index:]


def select_statistical_sample(dataset: List[Tuple[List[float], int]], sample_size: int) -> List[Tuple[List[float], int]]:
    """
    Seleciona uma amostra estatisticamente representativa do conjunto de dados.
    """
    if sample_size > len(dataset):
        sample_size = len(dataset)
    return random.sample(dataset, sample_size)


def main() -> None:
    """
    Função principal que coordena o carregamento dos dados,
    treinamento e validação do Perceptron.
    """
    random.seed(42)
    filepath = "amostras.csv"

    # Carrega e seleciona uma amostra representativa
    dataset = load_dataset(filepath)
    sample = select_statistical_sample(dataset, sample_size=30)

    # Divide em treino e validação
    training_data, validation_data = split_dataset(sample)

    # Inicializa e treina o Perceptron
    perceptron = Perceptron(num_inputs=6, learning_rate=0.005)
    perceptron.train(training_data, validation_data, max_epochs=100)


if __name__ == "__main__":
    main()
