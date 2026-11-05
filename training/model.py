"""La red: un embedding por usuario, uno por álbum, y el producto punto dice "cuánto le gusta"."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_users, n_albums, dim=32, lr=0.003):
    user_in = keras.Input(shape=(1,), name="user")
    album_in = keras.Input(shape=(1,), name="album")

    # Cada usuario y cada álbum se convierte en un vector de `dim` números
    user_vec = layers.Flatten()(layers.Embedding(n_users, dim, name="user_emb")(user_in))
    album_vec = layers.Flatten()(layers.Embedding(n_albums, dim, name="album_emb")(album_in))

    # Producto punto: si los vectores apuntan parecido, el número es alto
    score = layers.Dot(axes=1)([user_vec, album_vec])
    prob = layers.Activation("sigmoid")(score)   # lo pasa a 0..1 = prob. de que le guste

    model = keras.Model([user_in, album_in], prob)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model
