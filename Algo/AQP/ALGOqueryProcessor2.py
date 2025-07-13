import pandas as pd
import yfinance as yf
import customtkinter as tk
import queue
import threading
import time
import logging
import numpy as np

from Algo import ALGOdt4

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
                self.queryALGO(self.main.ALGOqueryQueue.iloc[0])
            time.sleep(0.1)

    def queryALGO(self, ALGOquery):
        ticker = ALGOquery["ticker"]
        duration = ALGOquery["duration"]
        dataSize = ALGOquery["dataSize"]
        ALGOqueryID = int(ALGOquery["ALGOqueryID"])

        successfulDownload = False
        attempts = 0
        maxAttempts = 10
        while not successfulDownload and attempts < maxAttempts:
            try:
                # Download the data
                tickerData = yf.download(
                    ticker,
                    group_by="Ticker",
                    period=duration,
                    interval="1m",
                    prepost=False,
                    repair=True,
                )

                # Reset the index to convert the datetime index to a column and get rid of timezone offest
                tickerData.index.tz_localize(None)
                tickerData.reset_index(inplace=True)

                # Keep only the 'Datetime' and 'Close' columns
                tickerData = tickerData[["Datetime", "Close", "Volume"]]
                if not tickerData.empty:
                    successfulDownload = True
            except Exception as e:
                print(f"ERROR: {e}")
                attempts += 1
        if not successfulDownload:
            print("FAILED TO DOWNLOAD DATA AFTER SEVERAL ATTEMPTS")

        # Extract the segment for close and volume
        tickerDataTimeframe = tickerData["Datetime"].to_numpy()
        tickerDataClose = tickerData["Close"].to_numpy()
        tickerDataVolume = tickerData["Volume"].to_numpy()

        # Compute the z-score for close and volume
        closeZScores = (4 * (tickerDataClose - np.mean(tickerDataClose))) / np.std(tickerDataClose)
        volumeZScores = (tickerDataVolume - np.mean(tickerDataVolume)) / np.std(tickerDataVolume)
        
        # Concats the 2 z-score sets
        tickerDataZScores = np.column_stack([tickerDataTimeframe, closeZScores, volumeZScores])

        # Algo predicts 15% of downloaded data length into the future
        predictionLen = tickerData.shape[0] // 5

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
        
        def queryLocalMachine():
            # Call algo
            algoSettings = ALGOdt4.Algo(
                int(dataSize),
                predictionLen,
                tickerDataZScores,
                self.main.progressQueue,
            )

            # Return result
            workerResultQueue.put(algoSettings.startPool())

        # Dataframe to store results from algo
        averageResult = pd.DataFrame()

        # Queue for results from worker
        workerResultQueue = queue.Queue()

        # Start local algo instance and wait for completion
        localMachineThread = threading.Thread(target=queryLocalMachine)
        localMachineThread.start()
        localMachineThread.join()

        # Number of workers. average result should have 100 columns (100 results)
        for result in range(1):
            averageResult = pd.concat([averageResult, workerResultQueue.get()], axis=1)

        # Turn on radiobutton
        self.AQradioButton.configure(text=ticker + " - " + duration, state="normal")

        # Remove AQ from ALGOqueryQueue
        self.main.ALGOqueryQueue = self.main.ALGOqueryQueue[1:]

        # Add AQ to history
        self.main.ALGOqueryHistory = pd.concat(
            [self.main.ALGOqueryHistory, ALGOquery.to_frame().T], ignore_index=True
        )

        # I should make this work at some point but I don't want to
        '''
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
        '''
            
        # Append AQ dataFrames for future use
        self.main.ALGOqueryResultHistory.append(averageResult)
        self.main.ALGOqueryDataHistory.append(tickerDataZScores[:, 1])

        self.AQradioButton.invoke()
