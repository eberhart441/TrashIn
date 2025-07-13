import customtkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd
import numpy as np
import yfinance as yf
import warnings

import multiprocessing as mp
import threading
import queue
import sys

from AutoTrade import ALGOat3
from Algo.AQP import ALGOqueryProcessor2

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class stdoutRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass

class App(tk.CTk):
    def __init__(self):
        super().__init__()

        def updateApp():
            try:
                self.clock += 1
                if self.clock % 60 == 0: # Triggers if a minute has passed - % means mod
                    # Refresh algo log
                    self.liveAlgoLogUpdates.delete("0.0", tk.END)
                    self.liveAlgoLogUpdates.insert(tk.END, open("algoLog.log").read())
                    self.liveAlgoLogUpdates.see(tk.END)

                    # Check if the stock marken is open. If not, theres no point of downloading new data
                    if ALGOat3.marketOpenCheck() == True:
                        print("msrkert is open")
                        # Update algoQueryDataHistory by downloading more data and exchanging it with current data
                        if len(self.ALGOqueryHistory) != 0:
                            for i in range(len(self.ALGOqueryHistory)):
                                print("index:", i)
                                #  Retreive a stock ticker form the query history
                                ticker = self.ALGOqueryHistory.iloc[i][0]
                                resultSize = self.ALGOqueryHistory.iloc[i][3]

                                successfulDownload = False
                                attempts = 0
                                maxAttempts = 10
                                while not successfulDownload and attempts < maxAttempts:
                                    try:
                                        # Download the data
                                        tickerData = yf.download(
                                            ticker,
                                            group_by="Ticker",
                                            period="7d",
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
                                if (not successfulDownload) or (len(tickerData) <= 390):
                                    print("FAILED TO DOWNLOAD DATA AFTER SEVERAL ATTEMPTS")

                                else:
                                    # Extract the segment for close and volume
                                    tickerDataTimeframe = tickerData["Datetime"].to_numpy()
                                    tickerDataClose = tickerData["Close"].to_numpy()
                                    tickerDataVolume = tickerData["Volume"].to_numpy()

                                    # Compute the z-score for close and volume
                                    closeZScores = (4 * (tickerDataClose - np.mean(tickerDataClose))) / np.std(tickerDataClose)
                                    volumeZScores = (tickerDataVolume - np.mean(tickerDataVolume)) / np.std(tickerDataVolume)
                                    
                                    # Concats the 2 z-score sets
                                    tickerDataZScores = np.column_stack([tickerDataTimeframe, closeZScores, volumeZScores])
                                    print("tickerDataZScores: ", tickerDataZScores)
                                    print(self.ALGOqueryDataHistory)

                                    # Find the first date in the history dataframe
                                    print(self.ALGOqueryDataHistory[i])
                                    firstDatetime = pd.to_datetime(self.ALGOqueryDataHistory[i][:, 0]).max()
                                    print(firstDatetime)

                                    # Filter the new data to only include rows after the last datetime in data2
                                    filteredDates = pd.to_datetime(tickerDataZScores[:, 0]) > firstDatetime
                                    tickerDataZScores = tickerDataZScores[filteredDates]

                                    self.ALGOqueryDataHistory.iloc[
                                        self.ALGOqueryHistory.iloc[:, 4]
                                    ] = tickerDataZScores

                                    print("hotdog")
                                    self.updateResultSize(resultSize)

                    else:
                        print("msrkert is closed")



                # Checks for progress in the queue
                while True:
                    progress = self.progressQueue.get_nowait()
                    if progress:
                        # Update the progress bar with the current value of self.progressValue
                        self.progressBar.set(progress)

            except queue.Empty:
                pass
            finally:
                self.after(1000, updateApp)

        # Declare int var to use for graph history radiobutton
        self.AQbuttonVar = tk.IntVar()

        self.ALGOqueryQueue = pd.DataFrame(
            columns=[
                "ticker",
                "duration",
                "dataSize",
                "resultSize",
            ]
        )
        self.ALGOqueryHistory = pd.DataFrame(
            columns=[
                "ticker",
                "duration",
                "dataSize",
                "resultSize",
            ]
        )

        self.ALGOqueryResultHistory = []
        self.ALGOqueryDataHistory = []
        self.ALGOqueryButtonHistory = []

        self.ALGOqueryID = 0

        # Declare progressBar and set it to 0
        self.progressBar = tk.CTkProgressBar(self, height=10)
        self.progressBar.grid(row=3, columnspan=3, sticky="ew")
        self.progressBar.set(0)

        # Shared queue of progress updates
        manager = mp.Manager()
        self.progressQueue = manager.Queue()

        # Initilize AQP
        AQthread = threading.Thread(target=lambda: ALGOqueryProcessor2.AQP(self))
        AQthread.start()

        # Initilize internal clock
        self.clock = 0

        self.configureWindow()
        self.declareGraphHistoryFrame()
        self.declareGraph()
        self.declareTabView()

        # Run function to check for progress updates inside other thread
        updateProgressBarThread = threading.Thread(target=updateApp)
        updateProgressBarThread.start()

    def configureWindow(self):
        # Set title and resizing properties
        self.title("BuyIn")
        self.resizable(True, True)

        # Set weight for expandable columns and rows
        self.grid_columnconfigure(1, weight=1)  # Graph column to expand horizontally
        self.grid_rowconfigure(0, weight=0)  # Graph row to expand vertically
        self.grid_rowconfigure(
            1, weight=1
        )  # The row for self.liveAlgoUpdates to expand vertically

        # Default apperence properties for app
        tk.set_appearance_mode("dark")
        tk.set_default_color_theme("blue")

    def declareGraphHistoryFrame(self):
        # Declare graph history frame
        self.graphHistoryFrame = tk.CTkScrollableFrame(
            self, label_text="BuyIn", width=225
        )
        self.graphHistoryFrame.grid(rowspan=2, row=0, column=0, sticky="nsew")
        self.graphHistoryFrame.columnconfigure(0, weight=1)

    def declareGraph(self):
        # Configure default graph settings
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)

        # Set the color of figure and axes backgrounds to match the GUI background
        self.plot.set_facecolor("#1d1e1e")

        # Shrink graph border
        self.plot.axes.get_xaxis().set_visible(False)
        self.plot.axes.get_yaxis().set_visible(False)
        self.fig.tight_layout(pad=0)

        self.plot.tick_params(labelcolor="white")
        self.plot.plot(color="#1f6aa5")

        # Make sure the graph expands in all directions
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2, sticky="nsew")

    def declareTabView(self):
        def declareStockPredictTab(self):
            # Declare 'Stock Predict' objects
            self.spTickerEntryBox = tk.CTkEntry(
                self.tabview.tab("Stock Predict"), placeholder_text="Ticker"
            )
            self.spTickerEntryBox.grid(row=0, column=0, padx=10, pady=(10, 0))
            self.spDurationBox = tk.CTkOptionMenu(
                self.tabview.tab("Stock Predict"),
                values=["1d", "5d", "7d"],
            )
            self.spDurationBox.grid(row=1, column=0, padx=10, pady=(10, 0))
            self.spDurationBox.set("5d")
            self.spButton = tk.CTkButton(
                self.tabview.tab("Stock Predict"),
                text="Find Matches",
                command=lambda: self.ALGOquery(
                    self.spTickerEntryBox.get(),
                    self.spDurationBox.get(),
                    self.spDataSizeSlider.get(),
                    self.spResultSizeSlider.get(),
                ),
            )
            self.spButton.grid(row=2, column=0, padx=10, pady=(10, 0))

            self.spDataSizeLabel = tk.CTkLabel(
                self.tabview.tab("Stock Predict"), text="Database Size - Result Size"
            )
            self.spDataSizeLabel.grid(row=0, column=1, padx=0, pady=(10, 0))

            self.spDataSizeSlider = tk.CTkSlider(
                self.tabview.tab("Stock Predict"),
                from_=16,
                to=400,  # 16 since its the number of cores
            )
            self.spDataSizeSlider.grid(row=1, column=1, padx=10, pady=(10, 0))
            self.spResultSizeSlider = tk.CTkSlider(
                self.tabview.tab("Stock Predict"),
                from_=1,
                to=100,
                command=self.updateResultSize,
            )
            self.spResultSizeSlider.grid(row=2, column=1, padx=10, pady=(10, 0))
            self.spResultSizeSlider.configure(number_of_steps=10)

        def declareAutotradingTab(self):
            def callAt():
                sys.stdout = stdoutRedirector(self.atLiveAlgorithmUpdates)
                ALGOat3.AutoTrade.repeat = True

                def runATthread():
                    AutoTrade = ALGOat3.AutoTrade(self)
                    AutoTrade.ALGOat

                # Start the thread and target the runATthread function.
                ALGOatThread = threading.Thread(target=runATthread)
                ALGOatThread.start()
                print("AT COMMAND SENT")

            def closeAt():
                print("Stopping Auto Trade...")
                ALGOat3.AutoTrade.repeat = False

            # Declare 'Stock Predict' objects
            self.atPathBox = tk.CTkEntry(
                self.tabview.tab("Auto Trade"), placeholder_text="Ticker"
            )
            self.atPathBox.grid(row=0, column=0, padx=10, pady=(10, 0))
            self.atPathBox.insert(
                tk.END,
                "C:/Users/Simon E/Documents/BuyIn/Resources/atProcesses.csv",
            )

            self.atStopButton = tk.CTkButton(
                self.tabview.tab("Auto Trade"),
                text="STOP",
                fg_color="#3B3B3B",
                command=closeAt,
            )
            self.atStopButton.grid(row=1, column=0, padx=10, pady=(10, 0))
            self.atStartButton = tk.CTkButton(
                self.tabview.tab("Auto Trade"),
                text="START",
                command=callAt,
            )
            self.atStartButton.grid(row=2, column=0, padx=10, pady=(10, 0))

            self.atLiveAlgorithmUpdates = tk.CTkTextbox(
                self.tabview.tab("Auto Trade"), width=200, height=100
            )
            self.atLiveAlgorithmUpdates.grid(
                row=0,
                rowspan=3,
                column=1,
                pady=(10, 0),
                sticky="ns",
            )

        # Declare tabview
        self.tabview = tk.CTkTabview(self, width=400, height=175)
        self.tabview.grid(row=0, column=2, padx=20, pady=(0, 20), sticky="nsew")

        # Add tabs to tabview
        self.tabview.add("Stock Predict")
        self.tabview.add("Auto Trade")

        declareStockPredictTab(self)
        declareAutotradingTab(self)

        self.liveAlgoLogUpdates = tk.CTkTextbox(self)
        self.liveAlgoLogUpdates.grid(
            row=1,
            column=2,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )
        self.liveAlgoLogUpdates.insert(tk.END, open("algoLog.log").read())
        self.liveAlgoLogUpdates.see(tk.END)

    def ALGOquery(self, ticker, duration, dataSize, resultSize):
        ALGOquery = pd.DataFrame(
            {
                "ticker": [ticker],
                "duration": [duration],
                "dataSize": [dataSize],
                "resultSize": [resultSize],
                "ALGOqueryID": [self.ALGOqueryID],
            }
        )
        self.ALGOqueryID += 1

        self.ALGOqueryQueue = pd.concat([self.ALGOqueryQueue, ALGOquery])

    def loadALGOquery(self):
        # Determine which radio button is currently clicked
        ALGOquery = self.ALGOqueryHistory[
            self.ALGOqueryHistory.iloc[:, 4] == self.AQbuttonVar.get()
        ]

        # Configure sp widget values based on selected radiobutton
        self.spTickerEntryBox.delete(0, tk.END)
        self.spTickerEntryBox.insert(0, ALGOquery.iloc[0, 0])
        self.spDurationBox.set(ALGOquery.iloc[0, 1])
        self.spDataSizeSlider.set(int(ALGOquery.iloc[0, 2]))
        self.spResultSizeSlider.set(int(ALGOquery.iloc[0, 3]))

        self.updateResultSize(ALGOquery.iloc[0, 3])
        
    def updateResultSize(self, resultSize):
        if self.ALGOqueryHistory.shape[0] != 0:
            # Determine which radio button is currently clicked
            index = self.AQbuttonVar.get()

            averageResult = self.ALGOqueryResultHistory[index]
            tickerDataZScores = self.ALGOqueryDataHistory[index]
            averageResult = averageResult.iloc[:, : int(resultSize)].mean(axis=1)

            # Graph data
            self.plot.cla()  # Clears data

            self.plot.plot(averageResult, color="white")

            self.plot.plot(tickerDataZScores, color="#1f6aa5", linewidth=3)

            # Check if tickerDataZScores is not empty and get the last values
            if tickerDataZScores.size > 0:
                last_y_value = tickerDataZScores[-1]
                last_x_value = len(tickerDataZScores) - 1
                last_y_value_averageResult = averageResult.iloc[-1]

                # Draw a horizontal dotted line at the last y value of tickerDataZScores
                self.plot.axhline(y=last_y_value, color="white", linestyle="dotted")
       
                # Draw a vertical dotted line at the last x value
                self.plot.axvline(x=last_x_value, color="white")

                # Draw a horizontal line at the x=len(tickerDataZScores) average result y value
                self.plot.axhline(
                    y=averageResult.iloc[len(tickerDataZScores) + 1], color="white", linestyle="dotted"
                )

                # Draw a horizontal dotted line at the last y value of averageResult
                self.plot.axhline(y=last_y_value_averageResult, color="white", linestyle="dotted")

            self.canvas.draw()


if __name__ == "__main__":
    # Initilize app
    app = App()
    app.mainloop()

# Crypto data downloaded 7/21/23