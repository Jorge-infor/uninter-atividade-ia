from __future__ import annotations
import csv
import random
from typing import List, Tuple


class Perceptron:
    """
    Implementação de um Perceptron (um único neurônio) treinado pela Regra Delta.

    Atributos:
        taxa_aprendizado (float): Taxa de aprendizado (η) usada para ajuste dos pesos.
        pesos (List[float]): Lista de pesos sinápticos do neurônio.
        bias (float): Valor de viés (bias) adicionado à ativação total.
    """

    def __init__(self, num_entradas: int, taxa_aprendizado: float = 0.1) -> None:
        self.taxa_aprendizado: float = taxa_aprendizado
        self.pesos: List[float] = [random.uniform(-1, 1) for _ in range(num_entradas)]
        self.bias: float = random.uniform(-1, 1)

    def ativacao(self, x: float) -> int:
        """Função de ativação degrau binária, retornando 1 ou -1."""
        return 1 if x >= 0 else -1

    def prever(self, entradas: List[float]) -> int:
        """Calcula a saída do perceptron para uma entrada fornecida."""
        soma_total = sum(w * e for w, e in zip(self.pesos, entradas)) + self.bias
        return self.ativacao(soma_total)

    def treinar(
        self,
        dados_treino: List[Tuple[List[float], int]],
        dados_validacao: List[Tuple[List[float], int]],
        max_epocas: int = 100
    ) -> None:
        """
        Treina o perceptron utilizando a Regra Delta.

        Args:
            dados_treino: Lista de tuplas (entradas, alvo) para treinamento.
            dados_validacao: Lista de tuplas (entradas, alvo) para validação.
            max_epocas: Número máximo de épocas de treinamento.
        """

        # Cria um arquivo CSV para registrar o processo de treinamento
        with open("treinamento.csv", mode="w", newline="", encoding="utf-8") as file:
            escritor = csv.writer(file, delimiter=";")
            escritor.writerow([
                "Época", "Amostra", "Entradas", "Esperado", "Saída", "Erro", "Pesos", "Bias"
            ])

            for epoca in range(1, max_epocas + 1):
                erro_total = 0
                print(f"\n=== Época {epoca} ===")

                for idx, (entradas, esperado) in enumerate(dados_treino, start=1):
                    saida = self.prever(entradas)
                    erro = esperado - saida
                    erro_total += abs(erro)

                    # Aplicando a Regra Delta: Δw = η * erro * x
                    for i in range(len(self.pesos)):
                        self.pesos[i] += self.taxa_aprendizado * erro * entradas[i]
                    self.bias += self.taxa_aprendizado * erro

                    # Arredonda pesos e bias para 2 casas decimais
                    pesos_arredondados = [round(p, 2) for p in self.pesos]
                    bias_arredondado = round(self.bias, 2)

                    # Mostra no terminal
                    print(
                        f"Amostra {idx}: Entradas={entradas}, Esperado={esperado}, "
                        f"Saída={saida}, Erro={erro}"
                    )
                    print(f"Pesos: {pesos_arredondados}")
                    print(f"Bias: {bias_arredondado}\n")

                    # Salva cada iteração no CSV
                    escritor.writerow([
                        epoca,
                        idx,
                        entradas,
                        esperado,
                        saida,
                        erro,
                        pesos_arredondados,
                        bias_arredondado
                    ])

                # Avalia desempenho com dados de validação
                acuracia = self.avaliar(dados_validacao)
                print(f"Acurácia na validação: {acuracia:.2f}%")

                # Condição de parada: 100% de acurácia
                if acuracia == 100.0:
                    print("\nTreinamento concluído com 100% de acurácia.")
                    break

        # Exibe pesos e bias finais
        pesos_finais = [round(p, 2) for p in self.pesos]
        bias_final = round(self.bias, 2)

        print("\n=== Treinamento Finalizado ===")
        print(f"Pesos finais: {pesos_finais}")
        print(f"Bias final: {bias_final}")

        # Exporta resultados finais para outro CSV
        with open("resultado.csv", mode="w", newline="", encoding="utf-8") as file_res:
            escritor_res = csv.writer(file_res, delimiter=";")
            escritor_res.writerow(["Pesos Finais", "Bias Final"])
            escritor_res.writerow([pesos_finais, bias_final])

    def avaliar(self, dados: List[Tuple[List[float], int]]) -> float:
        """Calcula a acurácia do perceptron em um conjunto de dados."""
        corretos = sum(1 for entradas, esperado in dados if self.prever(entradas) == esperado)
        return (corretos / len(dados)) * 100


def carregar_dados(caminho: str) -> List[Tuple[List[float], int]]:
    """
    Carrega o dataset a partir de um arquivo CSV.

    Formato esperado:
    Sensor 1;Sensor 2;...;Sensor 6;Controle Acionador
    """
    dados: List[Tuple[List[float], int]] = []
    with open(caminho, newline="", encoding="utf-8") as csvfile:
        leitor = csv.reader(csvfile, delimiter=";")
        next(leitor)  # Ignora o cabeçalho
        for linha in leitor:
            entradas = [float(x) for x in linha[:-1]]
            alvo = int(linha[-1])
            dados.append((entradas, alvo))
    return dados


def dividir_dados(
    dados: List[Tuple[List[float], int]],
    proporcao_treino: float = 0.7
) -> Tuple[List[Tuple[List[float], int]], List[Tuple[List[float], int]]]:
    """Embaralha e divide o conjunto de dados em subconjuntos de treino e validação."""
    random.shuffle(dados)
    limite = int(len(dados) * proporcao_treino)
    return dados[:limite], dados[limite:]


def amostra_estatistica(
    dados: List[Tuple[List[float], int]],
    tamanho_amostra: int
) -> List[Tuple[List[float], int]]:
    """
    Seleciona uma amostra representativa do dataset.
    Garante que as linhas sejam escolhidas aleatoriamente.
    """
    if tamanho_amostra > len(dados):
        tamanho_amostra = len(dados)
    return random.sample(dados, tamanho_amostra)


def main() -> None:
    """Rotina principal de carregamento, treinamento e validação do Perceptron."""
    random.seed(42)
    caminho = "amostras.csv"

    # Carrega e seleciona amostra estatisticamente representativa
    dados = carregar_dados(caminho)
    amostra = amostra_estatistica(dados, tamanho_amostra=30)

    # Divide a amostra entre treino e validação
    treino, validacao = dividir_dados(amostra)

    # Inicializa e treina o perceptron
    perceptron = Perceptron(num_entradas=6, taxa_aprendizado=0.005)
    perceptron.treinar(treino, validacao, max_epocas=100)


if __name__ == "__main__":
    main()
