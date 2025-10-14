from __future__ import annotations
import csv
import random
from typing import List, Tuple


class Perceptron:
    """
    Modelo de um único neurônio (Perceptron) treinado pela Regra Delta.

    O perceptron é a unidade fundamental de uma rede neural conexionista.
    Ele recebe várias entradas (sensores), pondera cada uma delas por pesos sinápticos,
    soma o resultado e aplica uma função de ativação binária que define a saída final.

    A Regra Delta é usada para ajustar os pesos e o bias com base no erro entre
    a saída esperada e a saída obtida.

    Atributos:
        learning_rate (float): Taxa de aprendizado (η), define o passo de ajuste dos pesos.
        weights (List[float]): Lista com os pesos sinápticos do neurônio.
        bias (float): Valor de bias (limiar de ativação).
    """

    def __init__(self, num_inputs: int, learning_rate: float = 0.1) -> None:
        """
        Inicializa o perceptron com pesos e bias aleatórios.

        Args:
            num_inputs (int): Número de entradas (sensores).
            learning_rate (float): Taxa de aprendizado.
        """
        self.learning_rate: float = learning_rate
        # Inicializa pesos e bias com valores aleatórios entre -1 e 1 (duas casas decimais)
        self.weights: List[float] = [round(random.uniform(-1, 1), 2) for _ in range(num_inputs)]
        self.bias: float = round(random.uniform(-1, 1), 2)

    def activation(self, x: float) -> int:
        """
        Função de ativação degrau binário.

        Retorna:
            1 se a soma ponderada for >= 0, caso contrário -1.
        """
        return 1 if x >= 0 else -1

    def predict(self, inputs: List[float]) -> int:
        """
        Calcula a saída do perceptron para uma amostra de entrada.

        Args:
            inputs (List[float]): Lista com os valores de entrada.

        Returns:
            int: Saída do perceptron (1 ou -1).
        """
        # Soma ponderada das entradas + bias
        total_activation = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation(total_activation)

    def train(
        self,
        training_data: List[Tuple[List[float], int]],
        validation_data: List[Tuple[List[float], int]],
        max_epochs: int = 100
    ) -> None:
        """
        Treina o perceptron usando a Regra Delta.

        Args:
            training_data (List[Tuple[List[float], int]]): Dados de treino (entradas e saídas esperadas).
            validation_data (List[Tuple[List[float], int]]): Dados de validação para avaliar o desempenho.
            max_epochs (int): Número máximo de épocas de treinamento.
        """
        for epoch in range(1, max_epochs + 1):
            total_error = 0
            print(f"\n=== Época {epoch} ===")

            # Percorre todas as amostras do conjunto de treino
            for idx, (inputs, expected) in enumerate(training_data, start=1):
                # Calcula a saída do perceptron
                output = self.predict(inputs)
                # Calcula o erro: diferença entre a saída esperada e a obtida
                error = expected - output
                total_error += abs(error)

                # Atualiza os pesos conforme a Regra Delta: Δw = η * erro * entrada
                for i in range(len(self.weights)):
                    delta_w = self.learning_rate * error * inputs[i]
                    self.weights[i] = round(self.weights[i] + delta_w, 2)

                # Atualiza o bias: Δb = η * erro
                delta_b = self.learning_rate * error
                self.bias = round(self.bias + delta_b, 2)

                # Exibe detalhes do treinamento para cada amostra
                print(f"Amostra {idx}: Entradas={inputs}, Esperado={expected}, "
                      f"Saída={output}, Erro={error}")
                print(f"Pesos: {self.weights}")
                print(f"Bias: {self.bias}\n")

            # Avalia a acurácia no conjunto de validação após cada época
            accuracy = self.evaluate(validation_data)
            print(f"Acurácia na validação: {accuracy:.2f}%")

            # Critério de parada: 100% de acurácia
            if accuracy == 100.0:
                print("\nTreinamento concluído com 100% de acurácia.")
                break

        # Exibe o resultado final do treinamento
        print("\n=== Treinamento Finalizado ===")
        print(f"Pesos finais: {self.weights}")
        print(f"Bias final: {self.bias}")

    def evaluate(self, dataset: List[Tuple[List[float], int]]) -> float:
        """
        Calcula a acurácia do perceptron em um conjunto de dados.

        Args:
            dataset (List[Tuple[List[float], int]]): Conjunto de dados (entradas e saídas esperadas).

        Returns:
            float: Acurácia percentual do perceptron.
        """
        correct = sum(1 for inputs, expected in dataset if self.predict(inputs) == expected)
        return (correct / len(dataset)) * 100


def load_dataset(filepath: str) -> List[Tuple[List[float], int]]:
    """
    Carrega o conjunto de amostras a partir de um arquivo CSV.

    O arquivo deve ter o formato:
    Sensor 1;Sensor 2;Sensor 3;Sensor 4;Sensor 5;Sensor 6;Controle Acionador

    Args:
        filepath (str): Caminho do arquivo CSV.

    Returns:
        List[Tuple[List[float], int]]: Lista de amostras, onde cada item contém
        (entradas, saída esperada).
    """
    dataset: List[Tuple[List[float], int]] = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)  # Pula o cabeçalho
        for row in reader:
            inputs = [float(x) for x in row[:-1]]  # Sensores
            target = int(row[-1])  # Saída esperada (1 ou -1)
            dataset.append((inputs, target))
    return dataset


def split_dataset(
    dataset: List[Tuple[List[float], int]], train_ratio: float = 0.7
) -> Tuple[List[Tuple[List[float], int]], List[Tuple[List[float], int]]]:
    """
    Embaralha e divide o conjunto de amostras em treino e validação.

    Args:
        dataset (List[Tuple[List[float], int]]): Lista de amostras completas.
        train_ratio (float): Proporção de dados usada para treinamento.

    Returns:
        Tuple[List[Tuple[List[float], int]], List[Tuple[List[float], int]]]:
        Dados de treino e validação.
    """
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_ratio)
    return dataset[:split_index], dataset[split_index:]


def select_statistical_sample(
    dataset: List[Tuple[List[float], int]], sample_size: int
) -> List[Tuple[List[float], int]]:
    """
    Seleciona uma amostra estatisticamente representativa do conjunto total.

    Em vez de pegar as primeiras linhas do arquivo, esta função escolhe aleatoriamente
    diferentes amostras para garantir que o treinamento envolva casos variados.

    Args:
        dataset (List[Tuple[List[float], int]]): Conjunto total de amostras.
        sample_size (int): Número de amostras desejadas.

    Returns:
        List[Tuple[List[float], int]]: Lista de amostras selecionadas.
    """
    if sample_size > len(dataset):
        sample_size = len(dataset)
    return random.sample(dataset, sample_size)


def main() -> None:
    """
    Função principal: executa todo o fluxo de carregamento, seleção,
    treinamento e validação do perceptron.
    """
    random.seed(42)  # Garante reprodutibilidade dos resultados
    filepath = "amostras.csv"  # Caminho do arquivo com os dados

    # Carrega as amostras do arquivo CSV
    dataset = load_dataset(filepath)

    # Seleciona uma amostra estatisticamente relevante
    sample = select_statistical_sample(dataset, sample_size=30)

    # Divide em conjuntos de treino (70%) e validação (30%)
    training_data, validation_data = split_dataset(sample)

    # Cria e treina o perceptron com taxa de aprendizado pequena (ajuste fino)
    perceptron = Perceptron(num_inputs=6, learning_rate=0.005)
    perceptron.train(training_data, validation_data, max_epochs=100)


if __name__ == "__main__":
    main()
