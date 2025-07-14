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
            print(pd.DataFrame(resultsList100).T)

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