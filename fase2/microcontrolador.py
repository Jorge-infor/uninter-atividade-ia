"""
Simulação do Neurônio Treinado em Microcontrolador

Este programa mantém o microcontrolador "ligado" e realiza leituras
periódicas de 6 sensores simulados. Para cada leitura, calcula a saída
do neurônio e exibe a decisão de manter o forno ativo ou desligar.
"""

import random
import time
from typing import List

# --- Função de ativação degrau ---
def ativacao(soma: float) -> int:
    """
    Função de ativação binária.
    Retorna:
        1 → Manter forno ativo
        -1 → Parar e desligar forno
    """
    return 1 if soma >= 0 else -1

# --- Função do neurônio ---
def neuronio(entradas: List[int]) -> int:
    """
    Calcula a saída do neurônio com base nas entradas fornecidas.
    """
    # === PESOS FINAIS DO TREINAMENTO ===
    pesos = [-0.03, -0.03, 0.15, 0.17, -0.11, 0.04]
    bias = -0.75

    soma_total = sum(p * e for p, e in zip(pesos, entradas)) + bias
    saida = ativacao(soma_total)
    return saida

# --- Função principal ---
def main() -> None:
    """
    Simula o microcontrolador em operação contínua.
    Leitura periódica dos sensores (mock) e cálculo do neurônio.
    """
    print("=== Simulação do Microcontrolador do Forno ===")
    print("Pressione Ctrl+C para encerrar a simulação.\n")

    try:
        while True:
            # --- Mock: gera 6 leituras inteiras de sensores entre -9 e 9 ---
            sensores = [random.randint(-9, 9) for _ in range(6)]

            # Calcula saída do neurônio
            saida = neuronio(sensores)

            # Exibe entradas e saída
            print(f"Sensores: {sensores}")
            if saida == 1:
                print("Saída do neurônio: 1 → Manter forno ATIVO 🔥\n")
            else:
                print("Saída do neurônio: -1 → Desligar forno e parar esteiras 🧊\n")

            # Intervalo de leitura (simula periodicidade)
            time.sleep(2)  # 2 segundos entre leituras

    except KeyboardInterrupt:
        # Simula desligamento do microcontrolador
        print("\nSimulação encerrada pelo usuário. Microcontrolador desligado.")


if __name__ == "__main__":
    main()
