import pandas as pd
import yfinance as yf
import customtkinter as tk
import queue
import threading
import time
import logging

import subprocess
import json

from Algo import ALGOdt3

# Set the logging level for yfinance to WARNING or higher
yfinanceLogger = logging.getLogger("yfinance")
yfinanceLogger.setLevel(logging.WARNING)


class AQP:
    def __init__(self, main):
        self.main = main
        self.AQcheck()

    def AQcheck(self):
        while True:
            if self.main.ALGOqueryQueue.shape[0] != 0:
                # Evaluate query
                self.retreveQueryData(self.main.ALGOqueryQueue.iloc[0])
            time.sleep(0.1)

    def retreveQueryData(self, ALGOquery):
        ticker = ALGOquery["ticker"]
        duration = ALGOquery["duration"]
        dataSize = ALGOquery["dataSize"]
        ALGOqueryID = int(ALGOquery["ALGOqueryID"])

        successfulDownload = False
        attempts = 0
        maxAttempts = 10
        while not successfulDownload and attempts < maxAttempts:
            try:
                tickerData = yf.download(
                    ticker,
                    group_by="Ticker",
                    period=duration,
                    interval="5m",
                    prepost=False,
                    repair=True,
                ).reset_index()["Close"]
                if not tickerData.empty:
                    successfulDownload = True
            except Exception as e:
                print(attempts)
                print(f"ERROR: {e}")
                attempts += 1
        if not successfulDownload:
            print("FAILED TO DOWNLOAD DATA AFTER SEVERAL ATTEMPTS")

        # Multiply data to end with "1"
        multiplyFactor = 1 / tickerData.iloc[-1]
        tickerDataMultiplied = tickerData.mul(multiplyFactor)

        # Algo predicts 20% of downloaded data length into the future
        predictionLen = tickerDataMultiplied.shape[0] / 5

        # Configure radiobutton text
        text = ticker + " - " + duration + "\nIn Queue"
        self.AQradioButton = tk.CTkRadioButton(
            self.main.graphHistoryFrame,
            text=text,
            variable=self.main.AQbuttonVar,
            value=ALGOqueryID,
            command=self.main.loadALGOquery,
            state="disabled",
        )
        self.AQradioButton.grid(row=ALGOqueryID, column=0, padx=10, pady=(0, 10))

        self.queryALGO(
            ticker,
            duration,
            dataSize,
            tickerData,
            tickerDataMultiplied,
            predictionLen,
            ALGOquery,
        )

    def queryALGO(
        self,
        ticker,
        duration,
        dataSize,
        tickerData,
        tickerDataMultiplied,
        predictionLen,
        ALGOquery,
    ):
        def queryNode0():
            # Call algo
            algoSettings = ALGOdt3.Algo(
                int(dataSize),
                predictionLen,
                tickerDataMultiplied,
                self.main.progressQueue,
            )

            # Return result
            nodeResultQueue.put(algoSettings.startPool())

        def queryNode1():
            ip = "192.168.50.234"

            # Serialize the data to a JSON string
            serialized_data = json.dumps(
                {
                    "dataSize": int(dataSize * 0.2),
                    "predictionLen": predictionLen,
                    "dataPercent": 0.65,
                    "tickerDataMultiplied": list(tickerDataMultiplied),
                }
            )

            # Modify the command to include your Linode program path and enclose the JSON data in single quotes
            command = [
                "ssh",
                f"{username}@{ip}",
                "python3",
                programPath,
                f"'{serialized_data}'",  # Enclose the JSON data in single quotes
            ]

            try:
                serverOutput = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if serverOutput.returncode == 0:
                    nodeResultQueue.put(
                        pd.read_json(serverOutput.stdout, orient="records")
                    )
                else:
                    print("Error executing the command with algo node 1.")
                    print("Error output:")
                    print(serverOutput.stderr)
            except subprocess.CalledProcessError:
                print(
                    "Unable to establish an SSH connection with algo node 1. Please check your credentials."
                )

        def queryNode2():
            ip = "192.168.50.235"

            # Serialize the data to a JSON string
            serialized_data = json.dumps(
                {
                    "dataSize": int(dataSize * 0.15),
                    "predictionLen": predictionLen,
                    "dataPercent": 0.65,
                    "tickerDataMultiplied": list(tickerDataMultiplied),
                }
            )

            # Modify the command to include your Linode program path and enclose the JSON data in single quotes
            command = [
                "ssh",
                f"{username}@{ip}",
                "python3",
                programPath,
                f"'{serialized_data}'",  # Enclose the JSON data in single quotes
            ]

            try:
                serverOutput = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if serverOutput.returncode == 0:
                    nodeResultQueue.put(
                        pd.read_json(serverOutput.stdout, orient="records")
                    )
                else:
                    print("Error executing the command with algo node 1.")
                    print("Error output:")
                    print(serverOutput.stderr)
            except subprocess.CalledProcessError:
                print(
                    "Unable to establish an SSH connection with algo node 1. Please check your credentials."
                )

        # Declare algo connection varibales
        username = "simon"
        programPath = "BuyIn/algoDeploy1.py"

        nodeResultQueue = queue.Queue()
        averageResult = pd.DataFrame()

        # Initiate queryNode threads
        queryNode0Thread = threading.Thread(target=queryNode0)
        queryNode1Thread = threading.Thread(target=queryNode1)
        queryNode2Thread = threading.Thread(target=queryNode2)

        # Start threads
        queryNode0Thread.start()
        queryNode1Thread.start()
        queryNode2Thread.start()

        # Wait for all nodes to complete
        queryNode0Thread.join()
        queryNode1Thread.join()
        queryNode2Thread.join()

        # Number of nodes. average result should have 100 columns (100 results)
        for result in range(3):
            averageResult = pd.concat([averageResult, nodeResultQueue.get()], axis=1)

        # Turn on radiobutton
        self.AQradioButton.configure(text=ticker + " - " + duration, state="normal")

        # Remove AQ from ALGOqueryQueue
        self.main.ALGOqueryQueue = self.main.ALGOqueryQueue[1:]

        # Add AQ to history
        self.main.ALGOqueryHistory = self.main.ALGOqueryHistory.append(
            ALGOquery, ignore_index=True
        )

        # Delete data after threshold to prevent unecissary space use
        self.main.ALGOqueryButtonHistory.append(self.AQradioButton)
        if len(self.main.ALGOqueryButtonHistory) > 50:
            self.main.ALGOqueryResultHistory[
                len(self.main.ALGOqueryResultHistory) - 50
            ] = 0
            self.main.ALGOqueryDataHistory[len(self.main.ALGOqueryDataHistory) - 50] = 0
            self.main.ALGOqueryButtonHistory.pop(0).destroy()

            self.main.ALGOqueryHistory = self.main.ALGOqueryHistory.drop(
                self.main.ALGOqueryHistory.index[0]
            )

        # Append AQ dataFrames for future use
        self.main.ALGOqueryResultHistory.append(averageResult)
        self.main.ALGOqueryDataHistory.append(tickerData)

        self.AQradioButton.invoke()
