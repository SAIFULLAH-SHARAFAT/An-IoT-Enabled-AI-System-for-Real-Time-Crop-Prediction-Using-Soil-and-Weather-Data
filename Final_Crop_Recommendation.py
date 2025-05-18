import os
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
import time
from pymodbus.client import ModbusSerialClient as ModbusClient  
from pymodbus.exceptions import ModbusException
from time import sleep
from datetime import datetime

# Load pre-trained scaler, label encoder, and model from local paths
scaler = joblib.load('/home/rasp499/Desktop/Agriculture/scaler.pkl')
label_encoder = joblib.load('/home/rasp499/Desktop/Agriculture/label_encoder.pkl')
model = joblib.load('/home/rasp499/Desktop/Agriculture/random_search_model.pkl')

# WeatherAPI configuration
WEATHER_API_KEY = "Hide for safety"  # Replace with your WeatherAPI key
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
LOCATION = "Dhaka"  

# ThingsBoard configuration
THINGSBOARD_HOST = "https://demo.thingsboard.io"  # Replace with your ThingsBoard host
ACCESS_TOKEN = "hidden"  # Replace with your device's access token
THINGSBOARD_API_URL = f"{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry"

# Configure Modbus client
client = ModbusClient(
    port="/dev/ttyUSB0",  # Adjust if needed for your system
    baudrate=4800,
    stopbits=1,
    bytesize=8,
    parity="N",
    timeout=1
)

# Function to return rainfall based on the month
def get_rainfall():
    month = time.localtime().tm_mon
    rainfall_values = {
        1: 9.4, 2: 36.2, 3: 155.3, 4: 375.6,
        5: 569.6, 6: 818.4, 7: 819.2, 8: 612.6,
        9: 535.9, 10: 223.9, 11: 30.4, 12: 9.4
    }
    return rainfall_values.get(month, 0)

# Function to fetch humidity data from WeatherAPI
def fetch_humidity():
    try:
        params = {"key": WEATHER_API_KEY, "q": LOCATION}
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        weather_data = response.json()
        humidity = weather_data.get("current", {}).get("humidity", None)
        if humidity is not None:
            print(f"Humidity retrieved: {humidity}%")
        return humidity
    except requests.RequestException as e:
        print(f"Failed to fetch humidity data: {e}")
        return None

# Function to read and scale sensor data
def read_sensor_data(humidity):
    try:
        # Corrected for pymodbus 3.x: `unit` -> `slave`
        result = client.read_holding_registers(0, 7, slave=1)  # Use slave=1 instead of unit=1

        if result.isError():
            print("Modbus error while reading registers.")
            return None, None

        print("Modbus data:", result.registers)

        rainfall = get_rainfall()

        # Extract EC (electrical conductivity) from result.registers[2] (µS/cm)
        ec = result.registers[2]  # µS/cm value to send to ThingsBoard

        # Construct the data dictionary for prediction (excluding EC)
        data = {
            "N": result.registers[4] * 2,  # Adjust according to sensor scaling
            "P": result.registers[5] * 2,
            "K": result.registers[6] * 2,
            "pH": result.registers[3] / 10.0,
            "Temp(°C)": result.registers[1] / 10.0,
            "Humidity(%)": humidity,  # Using fetched humidity
            "Moisture(%)": result.registers[0] / 10.0,  # Assuming moisture is read from a sensor
            "Rainfall(cm)": rainfall  # Using hardcoded rainfall values based on the month of BAMIS portal
        }

        df = pd.DataFrame([data])
        print("Processed sensor data:\n", df.to_string(index=False))

        # Return both the dataframe for prediction and the EC value for telemetry
        return df, ec

    except ModbusException as e:
        print(f"Modbus error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

# Function to send data to ThingsBoard with a timestamp to avoid overwriting
def send_to_thingsboard(data, crop_name, ec):
    try:
        # Get the current Unix timestamp (milliseconds)
        timestamp = int(time.time() * 1000)  # Time in milliseconds

        # Prepare telemetry data (add EC but do not include it in the model dataframe)
        telemetry_data = {
            "N": int(data["N"][0]),
            "P": int(data["P"][0]),
            "K": int(data["K"][0]),
            "Moisture": float(data["Moisture(%)"][0]),
            "Temperature": float(data["Temp(°C)"][0]),
            "pH": float(data["pH"][0]),
            "Humidity": float(data["Humidity(%)"][0]),
            "Rainfall": float(data["Rainfall(cm)"][0]),
            "RecommendedCrop": crop_name,
            "EC": ec  # Send EC value (µS/cm) directly to ThingsBoard
        }

        # Add the timestamp to the telemetry data
        telemetry_data["ts"] = timestamp

        # Send data to ThingsBoard
        response = requests.post(
            THINGSBOARD_API_URL,
            headers={"Content-Type": "application/json"},
            json=telemetry_data
        )

        if response.status_code == 200:
            print("Data sent to ThingsBoard successfully!")
        else:
            print(f"Failed to send data to ThingsBoard. Status Code: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Failed to send data to ThingsBoard: {e}")

# Main loop to read data, fetch weather humidity, predict, and send results to ThingsBoard
if client.connect():
    try:
        print("Waiting for 1 minute before starting data collection...")
        sleep(6)

        humidity = fetch_humidity()  # Fetch humidity from WeatherAPI

        if humidity is not None:
            data, ec = read_sensor_data(humidity)  # Get both data and EC value

            if data is not None:
                # Scale the data (excluding EC)
                scaled_data = scaler.transform(data)

                # Make prediction
                prediction = model.predict(scaled_data)

                # Decode the prediction if necessary
                if label_encoder is not None:
                    decoded_prediction = label_encoder.inverse_transform(prediction)
                    print("Prediction:", decoded_prediction[0])
                else:
                    decoded_prediction = prediction[0]
                    print("Prediction:", decoded_prediction)

                # Send data to ThingsBoard, including EC value
                send_to_thingsboard(data, decoded_prediction[0], ec)
        else:
            print("Humidity is not fetched from API.")

        print("Data collection, weather fetching, prediction, and ThingsBoard update completed. Exiting program.")
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        client.close()
        print("Modbus client disconnected.")
else:
    print("Failed to connect to Modbus client.")
