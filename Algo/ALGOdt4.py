import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
import heapq
import random
from datetime import datetime
from time import sleep

# Setting up logging with detailed formatting
logging.basicConfig(
    filename="algoLog.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class Algo:
    def __init__(self, dataSize, predictionLen, tickerDataZScores, progressQueue):
        """
        Initialize the Algo class with settings for processing financial data.

        Args:
            dataSize (int): The size of the data to process.
            predictionLen (int): The length of the prediction interval.
            tickerDataZScores (DataFrame): The pre-processed ticker data.
            progressQueue (Queue): A queue for progress updates.
        """
        self.settings = {
            "dataSize": dataSize,
            "predictionLen": predictionLen,
            "tickerDataZScores": tickerDataZScores,
            "progressQueue": progressQueue,
        }

        self.tickerDataPath = "C:/Users/simon/OneDrive/Documents/Resources/FRD500"
        self.tickerDataLen = len(self.settings["tickerDataZScores"])
        self.processSplitIndex = self.settings["dataSize"] // mp.cpu_count()

    def startPool(self):
        numProcesses = mp.cpu_count()
        manager = mp.Manager()
        progressList = manager.list([0] * numProcesses)
        resultsList = []

        try:
            try:
                files = os.listdir(self.tickerDataPath)
            except FileNotFoundError:
                logging.error(f"Directory not found: {self.tickerDataPath}")
                return np.array([])
            with mp.Pool(processes=numProcesses) as pool:
                results = [
                    pool.apply_async(self.algo, args=(processId, progressList, files))
                    for processId in range(numProcesses)
                ]

                # Progress bar logic for the first phase
                prevTotal = 0
                while any(result.ready() is False for result in results):
                    currentTotal = sum(progressList)
                    if currentTotal != prevTotal:
                        self.settings["progressQueue"].put(
                            currentTotal / self.settings["dataSize"]
                        )
                    prevTotal = currentTotal
                    sleep(0.1)

                for result in results:
                    processResult = result.get()
                    if processResult:
                        resultsList.extend(
                            processResult
                        )  # Append tuples to the resultsList

            # Reset the progress bar for the second phase
            self.settings["progressQueue"].put(0)
            progress100 = 0
            total100 = len(resultsList) if len(resultsList) < 100 else 100

            # Get the 100 smallest distances without sorting the entire list
            top100Results = heapq.nsmallest(100, resultsList, key=lambda x: x[0])

            resultsList100 = []
            for idx, (_, filePath, i) in enumerate(top100Results):
                # Load the file corresponding to filePath
                tickerData = pd.read_csv(
                    filePath,
                    usecols=["close"],
                    dtype={"close": "float"},
                    skiprows=range(1, i),
                    nrows=self.tickerDataLen + self.settings["predictionLen"],
                ).to_numpy()

                # Compute the z-score for close and volume
                tickerDataZScores = (4 * (tickerData - np.mean(tickerData))) / np.std(
                    tickerData
                ).ravel()

                resultsList100.append(tickerDataZScores)

                # Update progress for the second phase
                progress100 += 1
                self.settings["progressQueue"].put(progress100 / total100)

            resultsList100 = [np.squeeze(array) for array in resultsList100]

            # Convert to DataFrame and transpose
            return pd.DataFrame(resultsList100).T
        except Exception as e:
            logging.exception("An error occurred during startPool execution: " + str(e))
            return pd.DataFrame()

    def algo(self, processId, progressList, files):
        resultList = []  # Collects the results

        try:
            files = os.listdir(self.tickerDataPath)
        except FileNotFoundError:
            logging.error(f"Directory not found: {self.tickerDataPath}")
            return np.array([])

        tickerDataTime = self.settings["tickerDataZScores"][-1, 0]

        for x in range(self.processSplitIndex):
            filePath = os.path.join(
                self.tickerDataPath, files[x + self.processSplitIndex * processId]
            )
            try:
                tickerCache = pd.read_csv(
                    filePath,
                    usecols=["datetime", "close", "volume"],
                    dtype={"close": "float", "volume": "float"},
                    parse_dates=["datetime"],
                )
            except Exception as e:
                logging.error(f"Error reading {filePath}: {e}")
                continue

            # Efficient datetime conversion
            tickerCache["time"] = tickerCache["datetime"].dt.time
            tickerCacheTimes = tickerCache["time"].to_numpy()

            for rowNum in range(
                len(tickerCache)
                - self.tickerDataLen
                - int(self.settings["predictionLen"])
            ):
                if (
                    tickerCacheTimes[rowNum + self.tickerDataLen - 1]
                    == tickerDataTime.time()
                ):
                    # Extract the segment for close and volume
                    tickerCacheSegmentClose = tickerCache.iloc[rowNum : rowNum + self.tickerDataLen, 1].to_numpy()
                    tickerCacheSegmentVolume = tickerCache.iloc[rowNum : rowNum + self.tickerDataLen, 2].to_numpy()

                    # Compute the z-score for close and volume
                    closeZScores = (4 * (tickerCacheSegmentClose - np.mean(tickerCacheSegmentClose))) / np.std(tickerCacheSegmentClose)
                    volumeZScores = (tickerCacheSegmentVolume - np.mean(tickerCacheSegmentVolume)) / np.std(tickerCacheSegmentVolume)

                    # Create weighted z-scores using exponential weights and concatenate the z-score sets
                    weights = np.exp(np.linspace(1, 2, self.tickerDataLen))  # Exponential weights
                    segmentZScores = np.column_stack([closeZScores, volumeZScores]) * weights[:, np.newaxis]

                    # Compute the weighted distance
                    distance = np.linalg.norm(self.settings["tickerDataZScores"][:, [1, 2]] * weights[:, np.newaxis] - segmentZScores)
                    resultList.append((distance, filePath, rowNum))

            progressList[processId] += 1
        return resultList


class Backtester:
    def __init__(self, algo_instance, data, data_extended):
        self.algo = algo_instance
        self.data = data
        self.data_extended = data_extended  # Extended data for validation
        self.signal = None
        self.portfolio = None

    def run_backtest(self):
        self.signal = self.generate_signal()
        self.portfolio = self.simulate_trading()

    def generate_signal(self):
        prediction_df = self.algo.startPool()
        predicted_price = prediction_df.iloc[-1, -1]  # Last predicted value
        # Use 'iloc' for integer-location based indexing and 'loc' for label-based indexing
        actual_future_price = self.data_extended["close"].iloc[-1]
        # Generate buy (1) or sell (-1) signal
        self.signal = 1 if predicted_price > actual_future_price else -1
        return self.signal

    def simulate_trading(self):
        # Initialize portfolio
        portfolio = pd.DataFrame(index=[0], columns=["position", "cash", "profit"])
        portfolio.loc[0, "position"] = 0  # No position initially
        portfolio.loc[0, "cash"] = 1000000  # Starting cash
        portfolio.loc[0, "profit"] = 0  # No profit initially

        # Fake buy/sell at the last price of 'data'
        trade_price = self.data.iloc[-1]["close"]

        profit = 0

        # Determine the outcome based on the last price of 'data_extended'
        outcome_price = self.data_extended.iloc[-1]["close"]

        if np.isnan(outcome_price):
            print("Algo returned NaN outcome price")
        else:
            if self.signal == 1:  # Buy signal
                # If the price increased, profit is positive; otherwise, it's negative
                print("BUY")
                profit = outcome_price - trade_price
            elif self.signal == -1:  # Sell signal
                # If the price decreased, profit is positive; otherwise, it's negative+-
                print("SELL")
                profit = trade_price - outcome_price
            else:
                print("NO SIGNAL GENERATED")

            print(profit)

            # Update portfolio
            portfolio.loc[0, "profit"] = profit
            portfolio.loc[0, "cash"] += profit  # Adjust cash based on profit/loss

        return portfolio


if __name__ == "__main__":
    total_profit = 0  # Initialize total profit

    # Run the backtest 50 times
    for _ in range(50):
        # Select random file
        files = os.listdir("Resources/FRD500")
        if len(files) == 0:
            logging.error("No files found in directory.")
            continue

        fileIndex = random.randint(1, len(files) - 1)
        filePath = os.path.join("Resources/FRD500", files[fileIndex])

        # Read file and parse dates
        try:
            tickerData = pd.read_csv(
                filePath,
                usecols=["datetime", "close", "volume"],
                dtype={"close": "float", "volume": "float"},
                parse_dates=["datetime"],
            )
        except Exception as e:
            logging.error(f"Error reading {filePath}: {e}")

        # Isolate 2 days of data
        startIndex = random.randint(1, tickerData.shape[0] - 5000)  # 5000 is arbitrary
        tickerData["time"] = tickerData["datetime"].dt.time

        from datetime import datetime

        # Define the time as a string
        timeString = "15:59"

        # Convert the string to a datetime.time object
        timeObject = datetime.strptime(timeString, "%H:%M").time()

        for i in range(len(tickerData) - startIndex):
            if tickerData["time"][i + startIndex + 780 - 1] == timeObject:
                tickerDataNormal = tickerData.iloc[
                    i + startIndex : i + startIndex + 780
                ]  # 2d timeframe
                tickerDataExtended = tickerData.iloc[
                    i + startIndex : i + startIndex + 780 + 1
                ]  # 2d timeframe
                tickerDataTimes = tickerData["time"][
                    i + startIndex : i + startIndex + 780
                ].to_numpy()
                break

        # Compute the z-score for close and volume
        closeZScores = (
            4 * (tickerDataNormal["close"] - tickerDataNormal["close"].mean())
        ) / tickerDataNormal["close"].std()
        volumeZScores = (
            tickerDataNormal["volume"] - tickerDataNormal["volume"].mean()
        ) / tickerDataNormal["volume"].std()

        # Concatenate the z-scores
        tickerDataZScores = np.column_stack(
            [tickerDataTimes, closeZScores, volumeZScores]
        )

        # Create an instance of your Algo class
        algo_instance = Algo(
            dataSize=50,
            predictionLen=1,
            tickerDataZScores=tickerDataZScores,
            progressQueue=mp.Manager().Queue(),
        )

        # Create and run the Backtester instance with extended data for validation
        backtester = Backtester(algo_instance, tickerDataNormal, tickerDataExtended)
        backtester.run_backtest()

        # Update total profit
        total_profit += backtester.portfolio.loc[0, "profit"]
        print(f"Current profit: {total_profit}")

    print(f"Total profit after 50 runs: {total_profit}")
