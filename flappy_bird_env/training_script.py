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

# Definición de la arquitectura de la red neuronal
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

# Preprocesamiento de la observación del entorno
def preprocess(state):
    state = np.array(state[0])
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    state = transform(state)
    return state.view(-1, 128)

# Estrategia epsilon-greedy para la selección de acciones
def epsilon_greedy(model, state, eps):
    if random.random() > eps:
        with torch.no_grad():
            return model(state).argmax().item()
    else:
        return random.randint(0, 1)

# Función de entrenamiento del agente
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
            #print("***STATE: ", len(state))
            #print("***STATE[0]: ", len(state[0]))
            #print("***STATE[1]: ", len(state[1]))
            
            # Selección de acción epsilon-greedy
            state = torch.tensor(state, dtype=torch.float32, device=device).clone().detach()
            action = epsilon_greedy(dqn, state, eps)
            
            # Ejecución de la acción en el entorno
            next_state, reward, done, _, _ = env.step(action)
            #print("***NEXT-STATE: ", len(next_state))
            #print("***NEXT-STATE[0]: ", len(next_state[0]))
            #print("***NEXT-STATE[1]: ", len(next_state[1]))
            
            # Almacenamiento de la transición en el búfer de repetición
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) >= BATCH_SIZE:
                # Muestreo de un minibatch del búfer de repetición
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                #print("***MINIBATCH: ", len(minibatch[0]))
                states, actions, rewards, next_states, dones = zip(*minibatch)
                #print("***STATES: ", states[1])
                #print("***STATES: ", len(states))
                #print("***STATES[0]: ", len(states[0]))
                #print("***STATES[1]: ", len(states[1]))

                # Conversión de datos a tensores de PyTorch
                states = torch.tensor(states[0], dtype=torch.float32, device=device)
                #print("***STATES: ", len(states))
                #print("***STATES[0]: ", len(states[0]))
                #print("***STATES[1]: ", len(states[1]))
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states[0], dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.bool, device=device)

                # Cálculo de los valores Q actuales y futuros
                #current_q_values = dqn(states).gather(1, actions)
                current_q_values = dqn(states)
                next_q_values = target_dqn(next_states).max(1)[0].detach()
                #print("---NEXT-Q-VALUES: ",next_q_values)
                #print("++++NEXT-Q-VALUES / DONES SIZE: ",next_q_values.size(), dones.size())
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

# Función para que el agente juegue en el entorno
def play_game(dqn):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = epsilon_greedy(dqn, preprocess(state), eps=0)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print("Total reward:", total_reward)

# Función main en donde se inicializa y coordina el entrenamiento del agente
def main():
    input_dim = env.observation_space.shape[0]
    print("Observation space shape:", env.observation_space.shape[0])
    output_dim = env.action_space.n

    dqn = DQN(input_dim, output_dim).to(device)
    target_dqn = DQN(input_dim, output_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=BUFFER_SIZE)

    train_agent(dqn, target_dqn, optimizer, loss_fn, replay_buffer)
    play_game(dqn)

if __name__ == "__main__":
    main()