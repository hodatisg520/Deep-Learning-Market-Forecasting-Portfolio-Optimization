import main
payload = [[6000.0, 6050.0, 5950.0, 6020.0, 1000000, 6050.0, 50.0, 0.0, 0.0, 15.0, 0.05]] * 20
from main import buy_model, buy_scaler
prob = main.run_inference(buy_model, buy_scaler, payload, (20, 11))
print("Prob:", prob)
