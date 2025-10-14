"""
Implementação do Neurônio Treinado (Perceptron de 6 entradas)

Este programa simula o comportamento de um neurônio treinado
em um microcontrolador para controle de um forno industrial.

Objetivo:
    - Receber 6 valores de sensores (entradas).
    - Calcular a soma ponderada utilizando os pesos sinápticos treinados.
    - Aplicar a função de ativação degrau binária:
        Se soma >= 0 → saída = 1  (manter forno ativo)
        Se soma < 0 → saída = -1 (parar e desligar forno)
    - Imprimir a decisão simulando o comando ao atuador.
"""

from typing import List


def ativacao(soma: float) -> int:
    """
    Função de ativação degrau binária.

    Args:
        soma (float): Soma ponderada das entradas com seus pesos.

    Returns:
        int: 1 para manter o forno ativo, -1 para desligar.
    """
    return 1 if soma >= 0 else -1


def neuronio(entradas: List[float]) -> int:
    """
    Executa o cálculo da saída do neurônio com base nas entradas.

    Args:
        entradas (List[float]): Lista com os 6 valores dos sensores.

    Returns:
        int: Saída do neurônio (1 ou -1).
    """

    # === INSIRA AQUI OS PESOS FINAIS DO TREINAMENTO ===
    # Exemplo de pesos fictícios (substitua pelos seus valores reais)
    pesos = [-0.03, -0.03, 0.15, 0.17, -0.11, 0.04]
    bias = -0.75  # Exemplo de bias final

    # Calcula soma ponderada das entradas
    soma_total = sum(p * e for p, e in zip(pesos, entradas)) + bias

    # Aplica função de ativação
    saida = ativacao(soma_total)

    return saida


def main() -> None:
    """
    Função principal que simula a leitura dos sensores e
    a decisão do neurônio.
    """
    print("=== Simulação do Neurônio no Forno Industrial ===")
    print("Insira os valores dos 6 sensores (separados por espaço):")

    try:
        # Lê entradas do usuário
        entradas = list(map(float, input("Sensores: ").strip().split()))

        if len(entradas) != 6:
            print("Erro: É necessário inserir exatamente 6 valores de sensores.")
            return

        # Calcula a saída do neurônio
        saida = neuronio(entradas)

        # Exibe o resultado (simulação do atuador)
        if saida == 1:
            print("\nSaída do neurônio: 1 → Manter o forno ATIVO 🔥")
        else:
            print("\nSaída do neurônio: -1 → Desligar forno e parar esteiras 🧊")

    except ValueError:
        print("Erro: Insira apenas números válidos separados por espaço.")


if __name__ == "__main__":
    main()
