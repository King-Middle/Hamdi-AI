from tensorflow.keras.models import load_model
import numpy as np

model = load_model('/Users/ardaunal/Desktop/Hamdi/Forex/Hamdi_EURUSD2_Forex_2.0.keras')

# Let's try predicting with (1, 48, 12)
try:
    x_48 = np.random.rand(1, 48, 12)
    pred_48 = model.predict(x_48, verbose=0)
    print("Success with (1, 48, 12). Output:", pred_48)
except Exception as e:
    print("Failed with (1, 48, 12):", e)

# Let's try predicting with (1, 805, 12)
try:
    x_805 = np.random.rand(1, 805, 12)
    pred_805 = model.predict(x_805, verbose=0)
    print("Success with (1, 805, 12). Output:", pred_805)
except Exception as e:
    print("Failed with (1, 805, 12):", e)
