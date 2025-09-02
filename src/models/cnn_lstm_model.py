import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Flatten
from typing import Dict, List

class CNNLSTMMultiHead:
    """
    CNN-LSTM con tronco compartido + múltiples cabezas (actor/critic) por activo.

    Uso típico:
        model = CNNLSTMMultiHead(
            input_shape=(lookback, n_features),
            action_space=3,
            assets=["BTCUSDT","ETHUSDT","SOLUSDT"]
        )

        actor_btc = model.get_actor("BTCUSDT")
        critic_eth = model.get_critic("ETHUSDT")
    """

    def __init__(
        self,
        input_shape,
        action_space: int,
        assets: List[str],
        learning_rate: float = 2.5e-4
    ):
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate

        # Diccionarios de modelos por activo
        self.actor_heads: Dict[str, tf.keras.Model] = {}
        self.critic_heads: Dict[str, tf.keras.Model] = {}

        # Construir tronco compartido (backbone)
        self.backbone_input, self.backbone_output = self._build_backbone()

        # Crear heads iniciales
        for asset in assets:
            self._build_heads_for(asset)

    # ---------- Backbone compartido ----------
    def _build_backbone(self):
        """
        Devuelve: (inputs, z)
          - inputs: Input(shape=self.input_shape)
          - z: embedding compartido (tensor)
        """
        inputs = Input(shape=self.input_shape, name="backbone_input")

        x = Conv1D(
            filters=32, kernel_size=3, padding="same",
            activation="relu", name="shared_conv1d"
        )(inputs)

        x = MaxPooling1D(pool_size=2, name="shared_maxpool")(x)

        x = LSTM(
            units=32, return_sequences=False, name="shared_lstm"
        )(x)

        # Nota: LSTM con return_sequences=False ya es plano, añadimos Dense para compactar
        z = Dense(32, activation="relu", name="shared_dense")(x)

        return inputs, z

    # ---------- Heads por activo ----------
    def _build_heads_for(self, asset: str):
        """
        Crea (o recrea) las cabezas actor/critic para un activo
        reutilizando el backbone compartido.
        """
        asset = str(asset)
        # Actor head
        a = Dense(32, activation="relu", name=f"actor_{asset}_dense")(self.backbone_output)
        actor_out = Dense(
            self.action_space, activation="softmax", name=f"actor_{asset}_output"
        )(a)
        actor_model = Model(
            inputs=self.backbone_input, outputs=actor_out, name=f"actor_{asset}"
        )

        # Critic head
        c = Dense(32, activation="relu", name=f"critic_{asset}_dense")(self.backbone_output)
        critic_out = Dense(1, activation=None, name=f"critic_{asset}_output")(c)
        critic_model = Model(
            inputs=self.backbone_input, outputs=critic_out, name=f"critic_{asset}"
        )

        # (Opcional) compilar si usas .fit en algún momento
        # En PPO normalmente haces entrenamiento manual, así que no es obligatorio.
        # actor_model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss="categorical_crossentropy")
        # critic_model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss="mse")

        self.actor_heads[asset] = actor_model
        self.critic_heads[asset] = critic_model

    # ---------- API pública ----------
    def add_asset(self, asset: str):
        """Añade una nueva cabeza actor/critic para un activo nuevo."""
        if asset in self.actor_heads:
            return  # ya existe
        self._build_heads_for(asset)

    def list_assets(self) -> List[str]:
        """Lista de activos con cabeza creada."""
        return sorted(list(self.actor_heads.keys()))

    def get_actor(self, asset: str) -> tf.keras.Model:
        """Obtiene el actor específico del activo."""
        if asset not in self.actor_heads:
            raise KeyError(f"No existe actor para activo '{asset}'. Añádelo con add_asset().")
        return self.actor_heads[asset]

    def get_critic(self, asset: str) -> tf.keras.Model:
        """Obtiene el crítico específico del activo."""
        if asset not in self.critic_heads:
            raise KeyError(f"No existe critic para activo '{asset}'. Añádelo con add_asset().")
        return self.critic_heads[asset]

    # Retrocompatibilidad mínima si alguna parte del código espera .actor / .critic
    # Puedes fijarlas a un activo por defecto (p. ej., el primero) o levantar error explícito.
    @property
    def actor(self):
        raise AttributeError(
            "Este modelo es multi-head. Usa get_actor(asset) en lugar de .actor."
        )

    @property
    def critic(self):
        raise AttributeError(
            "Este modelo es multi-head. Usa get_critic(asset) en lugar de .critic."
        )
