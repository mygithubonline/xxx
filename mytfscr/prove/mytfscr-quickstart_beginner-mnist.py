import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#Costruisci il modello tf.keras.Sequential impilando i livelli. 
#Scegli un ottimizzatore e una funzione di perdita per l'allenamento:

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
  #tf.keras.layers.Dense(10, activation='softmax')
])

#Per ogni esempio il modello restituisce un vettore di punteggi 
#" logit " o " log-odds ", uno per ogni classe.

predictions = model(x_train[:1]).numpy()
predictions

#La funzione tf.nn.softmax converte questi logit in "probabilità" per ogni classe:

tf.nn.softmax(predictions).numpy()

#La perdita losses.SparseCategoricalCrossentropy prende un vettore di logit 
#e un indice True e restituisce una perdita scalare per ogni esempio.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Questa perdita è uguale alla probabilità logaritmica negativa della vera classe: 
#è zero se il modello è sicuro della classe corretta.
#Questo modello non addestrato fornisce probabilità vicine al casuale (1/10 per ogni classe), 
#quindi la perdita iniziale dovrebbe essere vicina a -tf.math.log(1/10) ~= 2.3 .

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#Il metodo Model.fit regola i parametri del modello per ridurre al minimo la perdita:
model.fit(x_train, y_train, epochs=5)

#Il metodo Model.evaluate controlla le prestazioni dei modelli, 
#solitamente su un " Validation-set " o " Test-set ".


model.evaluate(x_test,  y_test, verbose=2)

#Il classificatore di immagini è ora addestrato a una precisione del ~ 98% 
#su questo set di dati. Per saperne di più, leggi i tutorial di TensorFlow .
#Se vuoi che il tuo modello restituisca una probabilità, puoi avvolgere 
#il modello addestrato e attaccarvi il softmax:

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])