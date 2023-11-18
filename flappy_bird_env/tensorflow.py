import numpy as np
import tensorflow as tf
from keras import layers
from collections import deque
import random

from flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode="human")

# Crea la red neuronal convolucional y define la arquitectura de la misma
def create_network():
    model = tf.keras.Sequential([
        layers.Input(shape=(800, 576, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='linear')  # Capa de salida: (2) salidas para las dos posibles acciones del flappy-bird (saltar o no saltar)
    ])
    return model

# Crea una instancia del modelo de la red
Q_network = create_network()
Q_network_target = create_network()
Q_network_target.set_weights(Q_network.get_weights())

# Compila el modelo
Q_network.compile(optimizer='adam', loss='mse')  # loss'mse' = función de pérdida que se usa

# Hiperparámetros para DQN
gamma = 0.99  # Factor de descuento
epsilon = 0.1  # Factor de exploración
replay_memory_size = 50000
replay_memory = deque(maxlen=replay_memory_size)
batch_size = 32
update_target_frequency = 1000  # Actualizar la red objetivo cada 1000 pasos

# Función para seleccionar una acción basada en epsilon-greedy
def select_action(state):
    image_array = state[0]  # Extrae solo el array de imágenes
    print("-----STATE[0]: ", image_array)
    normalized_image_array = image_array / 255.0  # Normaliza las imágenes
    
    # Expande las dimensiones de la imagen para simular un lote de tamaño 1
    img_batch = np.expand_dims(normalized_image_array, axis=0) 
   
    # Redimensiona las imágenes a la forma deseada (800, 576, 3)
    resized_images = tf.image.resize(img_batch, (800, 576))

    if np.random.rand() < epsilon:
        return np.random.choice(2)  # Saltar (1) o no saltar (0) de forma aleatoria
    else:
        print("Forma de entrada esperada por el modelo:", Q_network.input_shape) 
        print(f"****Input image shape: {resized_images.shape}")
        
        Q_values = Q_network.predict(resized_images)
        print("Q_Values shape: ", Q_values.shape)
        return np.argmax(Q_values)

# Función para almacenar la transición en la memoria de repetición
def store_transition(state, action, reward, next_state, done):
    replay_memory.append((state, action, reward, next_state, done))

# Función para realizar una actualización de Q utilizando DQN
def update_Q():
    print("****Entre a Update_Q")
    if len(replay_memory) < batch_size:
        print("+++++Entre al if de replay_memory")
        return

    # Muestra un lote aleatorio de la memoria de repetición
    minibatch = np.array(random.sample(replay_memory, batch_size))

    # Extrae columnas de minibatch
    states = np.vstack(minibatch[:, 0])
    actions = minibatch[:, 1].astype(int)
    rewards = minibatch[:, 2]
    next_states = minibatch[:, 3]
    dones = minibatch[:, 4]

    print("***HOLAAAA")
    # Construye una lista de imágenes
    next_states_images = [state[0] for state in next_states]
    print("***HOLAAAA 222222")
    # Redimensiona las imágenes a la forma deseada (800, 576, 3)
    resized_images = [tf.image.resize(image, [800, 576]) for image in next_states_images]
    print("***Original image shape:", next_states_images[0].shape)
    print("***Resized image shape:", tf.image.resize(next_states_images[0], [800, 576]).shape)

    # Convierte las imágenes redimensionadas a un array numpy
    next_states_array = np.array([tf.image.resize(image, [800, 576]) / 255.0 for image in next_states_images])
    print("****Next states array shape:", next_states_array.shape)

    # Calcula el objetivo Q
    targets = rewards + gamma * np.max(Q_network_target.predict(next_states_array), axis=1) * (1 - dones)

    # Calcula los valores Q actuales
    Q_values = Q_network.predict(states)

    # Actualiza los valores Q para las acciones seleccionadas
    Q_values[np.arange(len(Q_values)), actions] = targets

    # Entrena la red
    Q_network.fit(states, Q_values, epochs=1, verbose=0)


# Actualiza la red objetivo
def update_target_network():
    Q_network_target.set_weights(Q_network.get_weights())

# Entrenamiento del agente con DQN
def training_agent_DQN(num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = select_action(state)
            next_state, reward, done, _ ,_= env.step(action)
            #print(f"***NEXT STATE: {next_state}")

            store_transition(state, action, reward, next_state, done)
            update_Q()

            episode_reward += reward
            state = next_state
            print("-----Llego hasta el final del while")
            if done:
                break

        if episode % update_target_frequency == 0:
            print("***Entré a episode - update")
            update_target_network()

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

if __name__ == '__main__':
    training_agent_DQN(100000)
    print("Después de la llamada a training_agent_DQN")
    Q_network.save("flappy_bird_DQN.h5")  # Guarda el modelo entrenado
    env.close()  # Cierra el entorno del flappy-bird