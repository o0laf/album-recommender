"""La red: un embedding por usuario, uno por álbum, y el producto punto dice "cuánto le gusta"."""
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore


class AlbumEmbeddingNetwork:

    @staticmethod
    def build(user_count, album_count, embedding_dimension=32, learning_rate=0.003):
        user_input = keras.Input(shape=(1,), name="user")
        album_input = keras.Input(shape=(1,), name="album")

        # Cada usuario y cada álbum se convierte en un vector de `dim` números
        user_vector = layers.Flatten()(layers.Embedding(user_count, embedding_dimension, name="user_emb")(user_input))
        album_vector = layers.Flatten()(layers.Embedding(album_count, embedding_dimension, name="album_emb")(album_input))

        # Producto punto: si los vectores apuntan parecido, el número es alto
        score = layers.Dot(axes=1)([user_vector, album_vector])
        probability = layers.Activation("sigmoid")(score)   # lo pasa a 0..1 = prob. de que le guste

        model = keras.Model([user_input, album_input], probability)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")
        return model
