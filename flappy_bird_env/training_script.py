import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torchvision import transforms

from flappy_bird_env import FlappyBirdEnv
env = FlappyBirdEnv(render_mode="human") # Se llama al enviroment del flappy-bird

# Hiperparámetros 
BATCH_SIZE = 800
BUFFER_SIZE = 10000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Definición de la arquitectura de la red neuronal
"""
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        print("Input dimension:", input_dim)
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1728, input_dim),
            nn.ReLU(),
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

"""
Realiza el preprocesamiento de la observación del entorno.

Parameters:
    state (numpy.ndarray): Observación del entorno.

Returns:
    torch.Tensor: Estado preprocesado.
"""
def preprocess(state):
    state = np.array(state[0])
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    state = transform(state)
    return state.view(-1, 128)

"""
Implementa la estrategia epsilon-greedy para la selección de acciones.

Parameters:
    model (torch.nn.Module): Modelo de la red neuronal.
    state (torch.Tensor): Estado actual del entorno.
    eps (float): Probabilidad de exploración.

Returns:
    int: Acción seleccionada.
"""
def epsilon_greedy(model, state, eps):
    if random.random() > eps:
        with torch.no_grad():
            return model(state).argmax().item()
    else:
        return random.randint(0, 1)



"""
Entrena al agente utilizando el algoritmo de Q-learning.

Parameters:
    dqn (DQN): Red neuronal principal.
    target_dqn (DQN): Red neuronal objetivo.
    optimizer (torch.optim.Optimizer): Optimizador para actualizar la red neuronal.
    loss_fn (torch.nn.Module): Función de pérdida.
    replay_buffer (deque): Búfer de repetición para almacenar transiciones.
    
Returns:
    None
"""
def train_agent(dqn, target_dqn, optimizer, loss_fn, replay_buffer):
    eps = EPS_START
    step_num = 0

    # Se define el número de episodios
    for episode in range(1000): 
        state = env.reset()
        state = torch.tensor(state[0], dtype=torch.float32, device=device)
        done = False

        # Bucle de entrenamiento del agente 
        while not done: 
            
            # Selección de acción epsilon-greedy
            state = torch.tensor(state, dtype=torch.float32, device=device).clone().detach()
            action = epsilon_greedy(dqn, state, eps)
            
            # Ejecución de la acción en el entorno
            next_state, reward, done, _, _ = env.step(action)
            
            # Almacenamiento de la transición en el búfer de repetición
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) >= BATCH_SIZE:
                # Muestreo de un minibatch del búfer de repetición
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                # Conversión de datos a tensores de PyTorch
                states = torch.tensor(states[0], dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states[0], dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.bool, device=device)

                # Cálculo de los valores Q actuales y futuros
                current_q_values = dqn(states)
                next_q_values = target_dqn(next_states).max(1)[0].detach()
                target_q_values = rewards + GAMMA * next_q_values * ~dones
                
                # Cálculo de la pérdida y actualización de la red neuronal
                loss = loss_fn(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            step_num += 1

        if episode % TARGET_UPDATE == 0:
            # Actualización del modelo objetivo
            target_dqn.load_state_dict(dqn.state_dict())

        if eps > EPS_END:
            # Reducción de la probabilidad de exploración epsilon
            eps -= (EPS_START - EPS_END) / EPS_DECAY

    print("Training complete.")

"""
Permite al agente jugar en el entorno después del entrenamiento.

Parameters:
    dqn (DQN): Red neuronal del agente.

Returns:
    None
"""
def play_game(dqn):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = epsilon_greedy(dqn, preprocess(state), eps=0)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print("Total reward:", total_reward)
    
    
"""
Función main en donde se inicializa y coordina el entrenamiento del agente.

Parameters:
    None

Returns:
    None
"""
def main():
    input_dim = env.observation_space.shape[0]
    print("Observation space shape:", env.observation_space.shape[0])
    output_dim = env.action_space.n

    dqn = DQN(input_dim, output_dim).to(device)
    target_dqn = DQN(input_dim, output_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=0.5)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=BUFFER_SIZE)

    train_agent(dqn, target_dqn, optimizer, loss_fn, replay_buffer)
    play_game(dqn)

if __name__ == "__main__":
    main()