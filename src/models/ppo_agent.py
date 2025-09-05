import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# IMPORTA EL MODELO MULTI-HEAD
from src.models.cnn_lstm_model import CNNLSTMMultiHead


class PPOAgent:
    """PPO Agent for cryptocurrency trading (multi-asset, multi-head compatible)"""
    
    def __init__(
        self,
        input_shape,
        action_space,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lam=0.95,
        kl_target=0.02,
        kl_cutoff_factor=3.0,
        adaptive_epsilon=True,
        use_lr_schedule=False,
        assets=None,
        default_asset=None
    ):
        # --- Inicialización básica ---
        self.input_shape = input_shape
        self.action_space = action_space
        self.initial_learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lam = lam
        self.use_lr_schedule = use_lr_schedule
        self.training_steps = 0
        
        # KL parameters
        self.kl_target = kl_target
        self.kl_cutoff_factor = kl_cutoff_factor
        self.adaptive_epsilon = adaptive_epsilon

        # Multi-head assets
        if assets is None or len(assets) == 0:
            assets = ["SINGLE"]
        self.assets = list(assets)
        self.default_asset = default_asset if default_asset else self.assets[0]

        # Crear modelo multi-head
        self.model = CNNLSTMMultiHead(
            input_shape=self.input_shape,
            action_space=self.action_space,
            assets=self.assets,
            learning_rate=learning_rate
        )

        # Actor/critic por defecto
        self._active_asset = self.default_asset
        self.actor = self.model.get_actor(self._active_asset)
        self.critic = self.model.get_critic(self._active_asset)
        
        # Diccionarios de optimizadores por activo
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        for asset in self.assets:
            if use_lr_schedule:
                actor_lr_schedule = ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=10000,
                    decay_rate=0.95,
                    staircase=True
                )
                critic_lr_schedule = PiecewiseConstantDecay(
                    boundaries=[5000, 15000, 30000],
                    values=[learning_rate * 3.0, learning_rate * 2.0, learning_rate * 1.0, learning_rate * 0.5]
                )
                self.actor_optimizers[asset] = Adam(learning_rate=actor_lr_schedule)
                self.critic_optimizers[asset] = Adam(learning_rate=critic_lr_schedule)
            else:
                self.actor_optimizers[asset] = Adam(learning_rate=0.0003)
                self.critic_optimizers[asset] = Adam(learning_rate=0.0005)


        # Inicializar memoria
        self.clear_memory()

    # ---------- Helpers multi-head ----------
    def set_active_asset(self, asset: str):
        if asset not in self.assets:
            raise KeyError(f"Asset '{asset}' no está en la lista de assets {self.assets}")
        self._active_asset = asset
        self.actor = self.model.get_actor(asset)
        self.critic = self.model.get_critic(asset)

    def get_actor(self, asset: str = None):
        asset = asset or self._active_asset
        return self.model.get_actor(asset)

    def get_critic(self, asset: str = None):
        asset = asset or self._active_asset
        return self.model.get_critic(asset)

    # ---------- Acción y memoria ----------
    def get_action(self, state, training=True, tau=0.1, asset: str = None):
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        use_asset = asset or self._active_asset
        actor = self.get_actor(use_asset)
        action_probs = actor.predict(state, verbose=0)[0]
        if training:
            action = np.random.choice(self.action_space, p=action_probs)
        else:
            action = np.argmax(action_probs)
        return action, action_probs

    def remember(self, state, action, reward, next_state, done, action_probs, asset: str = None):
        self.states.append(state.astype(np.float32) if isinstance(state, np.ndarray) else state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.astype(np.float32) if isinstance(next_state, np.ndarray) else next_state)
        self.dones.append(done)
        self.action_probs.append(action_probs)
        self.assets_memory.append(asset or self._active_asset)

    # ---------- PPO Helpers ----------
    def _calculate_kl_divergence(self, old_probs, new_probs):
        old_probs = tf.cast(old_probs, tf.float32)
        new_probs = tf.cast(new_probs, tf.float32)
        old_probs = tf.clip_by_value(old_probs, 1e-10, 1.0)
        new_probs = tf.clip_by_value(new_probs, 1e-10, 1.0)
        kl_div = tf.reduce_sum(old_probs * tf.math.log(old_probs / new_probs), axis=1)
        return tf.reduce_mean(kl_div)

    def _adjust_epsilon(self, kl_div):
        if kl_div > self.kl_cutoff_factor * self.kl_target:
            return max(self.epsilon * 0.5, 0.01)
        elif kl_div < self.kl_target / self.kl_cutoff_factor:
            return min(self.epsilon * 1.5, self.initial_epsilon)
        return self.epsilon

    def _compute_advantage(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = next_values[t] if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns

    def get_learning_rates(self, asset=None):
        if asset is None:
            asset = self._active_asset  # usar el activo actual por defecto

        if self.use_lr_schedule:
            actor_lr = self.actor_lr_schedules[asset](self.training_steps)
            critic_lr = self.critic_lr_schedules[asset](self.training_steps)
        else:
            actor_lr = self.actor_optimizers[asset].learning_rate
            critic_lr = self.critic_optimizers[asset].learning_rate

        return {"actor_lr": float(actor_lr), "critic_lr": float(critic_lr)}


    # ---------- Entrenamiento multi-activo ----------
    def train(self, batch_size=32, epochs=5, asset=None):
        """
        Entrena el agente PPO usando memoria acumulada.
        Si asset=None, se entrenan todos los activos (multi-head).
        """
        # Si no hay suficientes pasos almacenados, retornar vacíos
        if len(self.states) < batch_size:
            return {'actor_loss': [], 'critic_loss': [], 'total_loss': [], 'kl_div': []}

        # Convertir memoria a arrays numpy
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.array(self.dones)
        old_action_probs = np.array(self.action_probs, dtype=np.float32)
        assets_mem = np.array(self.assets_memory)

        history = {'actor_loss': [], 'critic_loss': [], 'total_loss': [], 'kl_div': []}

        # Lista de activos a entrenar
        assets_to_train = [asset] if asset else self.assets

        for asset in assets_to_train:
            # Seleccionar los indices de la memoria correspondientes al asset
            idxs = np.where(assets_mem == asset)[0]
            if idxs.size == 0:
                continue

            s, a, r, ns, d, oap = states[idxs], actions[idxs], rewards[idxs], next_states[idxs], dones[idxs], old_action_probs[idxs]

            # Establecer cabeza activa
            self.set_active_asset(asset)

            actor = self.get_actor(asset)
            critic = self.get_critic(asset)
            actor_opt = self.actor_optimizers[asset]
            critic_opt = self.critic_optimizers[asset]

            # Calcular valores y ventajas
            values = critic.predict(s, verbose=0).flatten()
            next_values = critic.predict(ns, verbose=0).flatten()
            advantages, returns = self._compute_advantage(r, values, next_values, d)
            actions_one_hot = tf.one_hot(a, self.action_space)

            # Entrenamiento por epochs
            for epoch in range(epochs):
                indices = np.arange(len(s))
                np.random.shuffle(indices)
                epoch_kl_divs = []

                for start_idx in range(0, len(indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(indices))
                    batch_idx = indices[start_idx:end_idx]

                    batch_states = tf.convert_to_tensor(s[batch_idx], dtype=tf.float32)
                    batch_actions = tf.gather(actions_one_hot, batch_idx)
                    batch_adv = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                    batch_ret = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
                    batch_old_probs = tf.convert_to_tensor(oap[batch_idx], dtype=tf.float32)

                    # -------- Actor --------
                    with tf.GradientTape() as tape:
                        current_probs = actor(batch_states, training=True)
                        kl_div = self._calculate_kl_divergence(batch_old_probs, current_probs)
                        epoch_kl_divs.append(float(kl_div))

                        ratio = tf.reduce_sum(current_probs * batch_actions, axis=1) / \
                                (tf.reduce_sum(batch_old_probs * batch_actions, axis=1) + 1e-8)

                        if self.adaptive_epsilon and len(epoch_kl_divs) > 0:
                            self.epsilon = self._adjust_epsilon(kl_div)

                        surrogate1 = ratio * batch_adv
                        surrogate2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_adv
                        actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                        entropy = -tf.reduce_mean(tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-8), axis=1))
                        actor_loss -= self.entropy_coef * entropy

                    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
                    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))

                    # -------- Critic --------
                    with tf.GradientTape() as tape_c:
                        value_pred = tf.reshape(critic(batch_states, training=True), [-1])
                        critic_loss = self.value_coef * tf.reduce_mean(tf.square(batch_ret - value_pred))

                    critic_grads = tape_c.gradient(critic_loss, critic.trainable_variables)
                    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))

                    # Guardar métricas
                    history['actor_loss'].append(float(actor_loss))
                    history['critic_loss'].append(float(critic_loss))
                    history['total_loss'].append(float(actor_loss + critic_loss))
                    history['kl_div'].append(float(kl_div))
                    self.training_steps += 1

                # Logging por epoch
                avg_kl = np.mean(epoch_kl_divs) if epoch_kl_divs else 0
                lr_info = self.get_learning_rates()
                print(f"[{asset}] Epoch {epoch+1}/{epochs}, Avg KL: {avg_kl:.6f}, "
                    f"Eps: {self.epsilon:.4f}, Actor LR: {lr_info['actor_lr']:.6f}, "
                    f"Critic LR: {lr_info['critic_lr']:.6f}")

        # Limpiar memoria al final
        self.clear_memory()

        return history


    # ---------- Memoria ----------
    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.action_probs = [], [], []
        self.assets_memory = []

    # ---------- Guardado y carga ----------
    def save_checkpoint(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        full_model_path = os.path.join(dir_path, "ppo_agent_full_model")
        self.model.save(full_model_path, include_optimizer=True)
        print(f"✅ Checkpoint completo guardado en: {full_model_path}")

    def load_checkpoint(self, dir_path: str):
        full_model_path = os.path.join(dir_path, "ppo_agent_full_model")
        if os.path.exists(full_model_path):
            self.model = load_model(full_model_path)
            self.actor = self.get_actor(self._active_asset)
            self.critic = self.get_critic(self._active_asset)
            print(f"✅ Checkpoint completo cargado desde: {full_model_path}")
        else:
            print(f"⚠️ No se encontró checkpoint en {full_model_path}")

    def save_weights(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for asset in self.assets:
            self.get_actor(asset).save_weights(os.path.join(dir_path, f"actor_{asset}.weights.h5"))
            self.get_critic(asset).save_weights(os.path.join(dir_path, f"critic_{asset}.weights.h5"))
        print(f"✅ Pesos guardados por activo en: {dir_path}")

    def load_weights(self, dir_path: str):
        for asset in self.assets:
            actor_w = os.path.join(dir_path, f"actor_{asset}.weights.h5")
            critic_w = os.path.join(dir_path, f"critic_{asset}.weights.h5")
            if os.path.exists(actor_w):
                self.get_actor(asset).load_weights(actor_w)
            if os.path.exists(critic_w):
                self.get_critic(asset).load_weights(critic_w)
        print(f"✅ Pesos cargados desde: {dir_path}")
