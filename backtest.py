import pandas as pd
import numpy as np
import os
import random
import logging
import time
from datetime import datetime
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional

# Import the existing algorithm
from Algo import ALGOdt4

# Simple logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class StrategyTester:
    def __init__(self, msft_file: str, frd500_path: str):
        self.msft_file = msft_file
        self.frd500_path = frd500_path
        self.msft_data = None
        self.valid_days = []
        
        # Test parameters
        self.consensus_thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
        self.context_hours = [8, 10, 12, 14, 16]
        self.data_sizes = [50, 100, 150, 200]
        
        print("Strategy Tester initialized")
        
    def load_data(self):
        """Load MSFT data and find valid trading days"""
        print("="*50)
        print("LOADING DATA")
        print("="*50)
        print(f"[DEBUG] Loading MSFT data from: {self.msft_file}")
        
        # Check file exists
        if not os.path.exists(self.msft_file):
            print(f"[ERROR] File not found: {self.msft_file}")
            return
        
        # Load data
        print(f"[DEBUG] Reading CSV file...")
        self.msft_data = pd.read_csv(self.msft_file)
        print(f"[DEBUG] Successfully loaded {len(self.msft_data)} rows")
        
        print(f"[DEBUG] Converting datetime column...")
        self.msft_data['datetime'] = pd.to_datetime(self.msft_data['datetime'])
        self.msft_data['date'] = self.msft_data['datetime'].dt.date
        self.msft_data['time'] = self.msft_data['datetime'].dt.time
        
        print(f"[DEBUG] Data sample:")
        print(self.msft_data.head())
        print(f"[DEBUG] Date range: {self.msft_data['date'].min()} to {self.msft_data['date'].max()}")
        
        # Find valid days (have 3:00-3:45 PM data + 16h context)
        print(f"[DEBUG] Finding valid trading days...")
        trading_start = datetime.strptime("15:00:00", "%H:%M:%S").time()
        trading_end = datetime.strptime("15:45:00", "%H:%M:%S").time()
        
        valid_count = 0
        for date in self.msft_data['date'].unique():
            day_data = self.msft_data[self.msft_data['date'] == date]
            
            has_trading_window = (day_data['time'].min() <= trading_start and 
                                day_data['time'].max() >= trading_end)
            has_context = day_data.index[0] >= 16 * 60  # 16h context available
            
            if has_trading_window and has_context:
                self.valid_days.append(date)
                valid_count += 1
                if valid_count <= 5:  # Show first 5
                    print(f"[DEBUG] Valid day: {date}")
        
        print(f"[SUCCESS] Found {len(self.valid_days)} valid trading days")
        
        if len(self.valid_days) == 0:
            print(f"[ERROR] No valid trading days found!")
            print(f"[DEBUG] Sample times - min: {self.msft_data['time'].min()}, max: {self.msft_data['time'].max()}")
        
    def prepare_context_data(self, end_idx: int, context_hours: int) -> np.ndarray:
        """Prepare context data for algorithm"""
        print(f"[DEBUG] Preparing context data: end_idx={end_idx}, context_hours={context_hours}")
        
        context_minutes = context_hours * 60
        start_idx = end_idx - context_minutes + 1
        
        print(f"[DEBUG] Context window: {start_idx} to {end_idx} (total: {context_minutes} minutes)")
        
        data = self.msft_data.iloc[start_idx:end_idx + 1].copy()
        print(f"[DEBUG] Context data shape: {data.shape}")
        
        # Calculate z-scores
        close_mean, close_std = data['close'].mean(), data['close'].std()
        volume_mean, volume_std = data['volume'].mean(), data['volume'].std()
        
        print(f"[DEBUG] Close stats: mean={close_mean:.2f}, std={close_std:.2f}")
        print(f"[DEBUG] Volume stats: mean={volume_mean:.0f}, std={volume_std:.0f}")
        
        if close_std == 0: 
            close_std = 1
            print(f"[WARNING] Close std was 0, setting to 1")
        if volume_std == 0: 
            volume_std = 1
            print(f"[WARNING] Volume std was 0, setting to 1")
        
        close_z = 4 * (data['close'] - close_mean) / close_std
        volume_z = (data['volume'] - volume_mean) / volume_std
        
        print(f"[DEBUG] Z-score ranges: close [{close_z.min():.2f}, {close_z.max():.2f}], volume [{volume_z.min():.2f}, {volume_z.max():.2f}]")
        
        # Create array with datetime objects (for algorithm compatibility)
        print(f"[DEBUG] Converting to algorithm format...")
        result = np.empty((len(data), 3), dtype=object)
        
        datetime_objects = []
        for i, dt in enumerate(data['datetime']):
            if hasattr(dt, 'to_pydatetime'):
                dt_obj = dt.to_pydatetime()
                datetime_objects.append(dt_obj)
            else:
                datetime_objects.append(dt)
                
        result[:, 0] = datetime_objects
        result[:, 1] = close_z.values.astype(float)
        result[:, 2] = volume_z.values.astype(float)
        
        # Show sample data
        print(f"[DEBUG] Sample context data (first 3 rows):")
        for i in range(min(3, len(result))):
            dt = result[i, 0]
            time_str = dt.strftime("%H:%M:%S") if hasattr(dt, 'strftime') else str(dt)
            print(f"  Row {i}: {time_str}, close_z={float(result[i,1]):.2f}, vol_z={float(result[i,2]):.2f}")
        
        # Show last time (what algorithm will match)
        last_dt = result[-1, 0]
        if hasattr(last_dt, 'time'):
            print(f"[DEBUG] Algorithm will look for patterns ending at: {last_dt.time()}")
        
        return result
    
    def run_algorithm(self, context_data: np.ndarray, data_size: int) -> Tuple[pd.DataFrame, float]:
        """Run algorithm and return results + execution time"""
        print(f"[DEBUG] Running algorithm with data_size={data_size}")
        print(f"[DEBUG] Context data shape: {context_data.shape}")
        
        # Check FRD500 path
        if not os.path.exists(self.frd500_path):
            print(f"[ERROR] FRD500 path not found: {self.frd500_path}")
            return pd.DataFrame(), 999.0
        
        frd_files = os.listdir(self.frd500_path)
        print(f"[DEBUG] FRD500 directory has {len(frd_files)} files")
        
        if len(frd_files) == 0:
            print(f"[ERROR] No files in FRD500 directory")
            return pd.DataFrame(), 999.0
            
        # Show sample FRD500 file
        try:
            sample_file = os.path.join(self.frd500_path, frd_files[0])
            sample_data = pd.read_csv(sample_file, nrows=3)
            print(f"[DEBUG] Sample FRD500 file ({frd_files[0]}):")
            print(sample_data)
            
            # Test datetime parsing on the sample file
            print(f"[DEBUG] Testing datetime parsing on sample file...")
            test_data = pd.read_csv(sample_file, nrows=10, parse_dates=['datetime'])
            print(f"[DEBUG] Datetime column type after parsing: {test_data['datetime'].dtype}")
            if test_data['datetime'].dtype == 'object':
                print(f"[WARNING] Datetime parsing failed - trying manual conversion...")
                test_data['datetime'] = pd.to_datetime(test_data['datetime'])
                print(f"[DEBUG] After manual conversion: {test_data['datetime'].dtype}")
                
        except Exception as e:
            print(f"[WARNING] Could not read sample FRD500 file: {e}")
        
        start_time = time.time()
        
        try:
            print(f"[DEBUG] Creating ALGOdt4.Algo instance...")
            
            # Create a simple progress queue to avoid multiprocessing issues
            import queue
            simple_queue = queue.Queue()
            
            algo = ALGOdt4.Algo(
                dataSize=data_size,
                predictionLen=30,
                tickerDataZScores=context_data,
                progressQueue=simple_queue  # Use simple queue instead of mp.Manager().Queue()
            )
            
            # Set correct path
            print(f"[DEBUG] Setting algorithm path to: {self.frd500_path}")
            algo.tickerDataPath = self.frd500_path
            
            print(f"[DEBUG] Running algo.startPool()...")
            results = algo.startPool()
            execution_time = time.time() - start_time
            
            print(f"[DEBUG] Algorithm completed in {execution_time:.2f}s")
            print(f"[DEBUG] Raw results type: {type(results)}")
            
            if isinstance(results, np.ndarray):
                print(f"[DEBUG] Converting numpy array to DataFrame")
                results = pd.DataFrame(results)
            elif results is None:
                print(f"[DEBUG] Algorithm returned None, creating empty DataFrame")
                results = pd.DataFrame()
                
            print(f"[DEBUG] Final results shape: {results.shape}")
            
            if results.empty:
                print(f"[WARNING] Algorithm returned empty results!")
                print(f"[DEBUG] This could mean:")
                print(f"  - No matching patterns found in FRD500 database")
                print(f"  - Time matching failed (algorithm looking for wrong time)")
                print(f"  - Algorithm parameters too restrictive")
            else:
                print(f"[SUCCESS] Algorithm found {results.shape[1]} patterns")
                if results.shape[0] > 0 and results.shape[1] > 0:
                    print(f"[DEBUG] Sample prediction values: {results.iloc[-1, :5].values if results.shape[1] >= 5 else results.iloc[-1, :].values}")
                
            return results, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"[ERROR] Algorithm failed after {execution_time:.2f}s: {e}")
            
            # Print full traceback for debugging
            import traceback
            print(f"[DEBUG] Full error traceback:")
            traceback.print_exc()
            
            # Check if it's the datetime parsing error
            if "dt accessor" in str(e) or "datetimelike" in str(e):
                print(f"[DEBUG] This is the datetime parsing error in ALGOdt4.py")
                print(f"[DEBUG] The FRD500 CSV files aren't being parsed with proper datetime columns")
                print(f"[DEBUG] We need to patch the ALGOdt4.py file or fix the CSV reading")
            
            return pd.DataFrame(), 999.0
    
    def calculate_consensus(self, results: pd.DataFrame) -> Tuple[str, float]:
        """Calculate consensus from algorithm results"""
        print(f"[DEBUG] Calculating consensus from results shape: {results.shape}")
        
        if results.empty:
            print(f"[DEBUG] Results are empty, returning NO_SIGNAL")
            return "NO_SIGNAL", 0.0
            
        # Get predictions (last row represents forward movement)
        if len(results) > 0:
            predictions = results.iloc[-1].values
            print(f"[DEBUG] Got {len(predictions)} predictions from last row")
            print(f"[DEBUG] Sample predictions: {predictions[:10] if len(predictions) >= 10 else predictions}")
        else:
            print(f"[DEBUG] Results DataFrame has no rows")
            return "NO_SIGNAL", 0.0
        
        if len(predictions) == 0:
            print(f"[DEBUG] No predictions available")
            return "NO_SIGNAL", 0.0
            
        up_votes = np.sum(predictions > 0)
        down_votes = np.sum(predictions < 0)
        zero_votes = np.sum(predictions == 0)
        total_votes = len(predictions)
        
        print(f"[DEBUG] Vote breakdown: UP={up_votes}, DOWN={down_votes}, ZERO={zero_votes}, TOTAL={total_votes}")
        
        up_consensus = up_votes / total_votes
        down_consensus = down_votes / total_votes
        
        print(f"[DEBUG] Consensus: UP={up_consensus:.2%}, DOWN={down_consensus:.2%}")
        
        if up_consensus > down_consensus:
            signal = "LONG"
            consensus = up_consensus
        else:
            signal = "SHORT"
            consensus = down_consensus
            
        print(f"[DEBUG] Final signal: {signal} with {consensus:.2%} consensus")
        return signal, consensus
    
    def simulate_trade(self, entry_idx: int, signal: str, threshold: float) -> Dict:
        """Simulate a trade with given parameters"""
        trade = {
            'entry_time': self.msft_data.iloc[entry_idx]['datetime'],
            'entry_price': self.msft_data.iloc[entry_idx]['close'],
            'signal': signal,
            'threshold': threshold,
            'exit_time': None,
            'exit_price': None,
            'return_pct': 0.0,
            'duration_minutes': 0,
            'exit_reason': 'NO_EXIT'
        }
        
        # Look ahead up to 45 minutes or end of data
        max_idx = min(entry_idx + 45, len(self.msft_data) - 1)
        
        # Check every 5 minutes (for performance)
        for check_idx in range(entry_idx + 5, max_idx + 1, 5):
            if check_idx >= 12 * 60:  # Ensure context available
                # Re-evaluate consensus
                context_data = self.prepare_context_data(check_idx, 12)
                results, _ = self.run_algorithm(context_data, 100)
                _, consensus = self.calculate_consensus(results)
                
                # Exit if consensus drops below threshold
                if consensus < threshold:
                    trade.update({
                        'exit_time': self.msft_data.iloc[check_idx]['datetime'],
                        'exit_price': self.msft_data.iloc[check_idx]['close'],
                        'duration_minutes': check_idx - entry_idx,
                        'exit_reason': f'CONSENSUS_DROP_{consensus:.2f}'
                    })
                    break
        
        # Force exit if no early exit
        if trade['exit_price'] is None:
            trade.update({
                'exit_time': self.msft_data.iloc[max_idx]['datetime'],
                'exit_price': self.msft_data.iloc[max_idx]['close'],
                'duration_minutes': max_idx - entry_idx,
                'exit_reason': 'TIME_LIMIT'
            })
        
        # Calculate return
        if signal == "LONG":
            trade['return_pct'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
        else:  # SHORT
            trade['return_pct'] = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100
        
        return trade
    
    def test_speed(self, num_tests: int = 5):  # Reduce to 5 tests for faster debugging
        """Test algorithm speed with different data sizes"""
        print(f"\n" + "="*50)
        print("SPEED TEST")
        print("="*50)
        
        test_days = random.sample(self.valid_days, min(num_tests, len(self.valid_days)))
        print(f"[DEBUG] Selected {len(test_days)} test days: {test_days}")
        
        results = []
        
        for data_size in [100]:  # Just test one data size for now
            print(f"\n[DEBUG] Testing data_size = {data_size}")
            times = []
            successful_tests = 0
            
            for i, date in enumerate(test_days):
                print(f"[DEBUG] Speed test {i+1}/{len(test_days)} for data_size {data_size} on {date}")
                
                try:
                    # Get random time during trading window
                    day_data = self.msft_data[self.msft_data['date'] == date]
                    trading_indices = day_data[
                        (day_data['time'] >= datetime.strptime("15:00:00", "%H:%M:%S").time()) &
                        (day_data['time'] <= datetime.strptime("15:30:00", "%H:%M:%S").time())
                    ].index
                    
                    print(f"[DEBUG] Found {len(trading_indices)} trading window indices")
                    
                    if len(trading_indices) > 0:
                        test_idx = random.choice(trading_indices)
                        print(f"[DEBUG] Selected index {test_idx}")
                        
                        context_data = self.prepare_context_data(test_idx, 12)
                        results_df, exec_time = self.run_algorithm(context_data, data_size)
                        
                        # Check what the algorithm found
                        signal, consensus = self.calculate_consensus(results_df)
                        print(f"[DEBUG] Signal: {signal}, Consensus: {consensus:.2%}")
                        
                        if exec_time < 999:  # Successful
                            times.append(exec_time)
                            successful_tests += 1
                            print(f"[SUCCESS] Test completed in {exec_time:.2f}s")
                        else:
                            print(f"[WARNING] Test failed")
                    else:
                        print(f"[WARNING] No trading window data for {date}")
                        
                except Exception as e:
                    print(f"[ERROR] Speed test failed for {date}: {e}")
                    continue
            
            avg_time = np.mean(times) if times else 999.0
            results.append({'data_size': data_size, 'avg_time': avg_time, 'tests': len(times)})
            print(f"[RESULT] Data size {data_size}: Average time = {avg_time:.2f}s ({successful_tests}/{len(test_days)} successful)")
        
        return results
    
    def test_consensus_thresholds(self, num_tests: int = 20):
        """Test different consensus thresholds"""
        print(f"\n=== CONSENSUS THRESHOLD TEST ===")
        
        test_days = random.sample(self.valid_days, min(num_tests, len(self.valid_days)))
        results = []
        
        for threshold in self.consensus_thresholds:
            trades = []
            
            print(f"Testing threshold = {threshold:.0%}")
            
            for date in test_days:
                # Test one trade per day
                day_data = self.msft_data[self.msft_data['date'] == date]
                trading_indices = day_data[
                    (day_data['time'] >= datetime.strptime("15:00:00", "%H:%M:%S").time()) &
                    (day_data['time'] <= datetime.strptime("15:10:00", "%H:%M:%S").time())
                ].index
                
                if len(trading_indices) > 0:
                    entry_idx = random.choice(trading_indices)
                    
                    # Get signal
                    context_data = self.prepare_context_data(entry_idx, 12)
                    algo_results, _ = self.run_algorithm(context_data, 100)
                    signal, consensus = self.calculate_consensus(algo_results)
                    
                    # Enter trade if consensus meets threshold
                    if signal in ["LONG", "SHORT"] and consensus >= threshold:
                        trade = self.simulate_trade(entry_idx, signal, threshold)
                        trade['consensus'] = consensus
                        trades.append(trade)
            
            # Calculate metrics
            if trades:
                df = pd.DataFrame(trades)
                win_rate = len(df[df['return_pct'] > 0]) / len(df)
                avg_return = df['return_pct'].mean()
                total_return = df['return_pct'].sum()
                
                results.append({
                    'threshold': threshold,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_return': total_return
                })
                
                print(f"  {len(trades)} trades, {win_rate:.1%} win rate, {avg_return:.2f}% avg return")
            else:
                results.append({
                    'threshold': threshold,
                    'trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'total_return': 0
                })
                print(f"  No trades")
        
        return results
    
    def test_context_windows(self, num_tests: int = 15):
        """Test different context window sizes"""
        print(f"\n=== CONTEXT WINDOW TEST ===")
        
        test_days = random.sample(self.valid_days, min(num_tests, len(self.valid_days)))
        results = []
        
        for context_h in self.context_hours:
            trades = []
            times = []
            
            print(f"Testing context = {context_h} hours")
            
            for date in test_days:
                day_data = self.msft_data[self.msft_data['date'] == date]
                trading_indices = day_data[
                    (day_data['time'] >= datetime.strptime("15:00:00", "%H:%M:%S").time()) &
                    (day_data['time'] <= datetime.strptime("15:10:00", "%H:%M:%S").time())
                ].index
                
                if len(trading_indices) > 0:
                    entry_idx = random.choice(trading_indices)
                    
                    # Check if enough context available
                    if entry_idx >= context_h * 60:
                        context_data = self.prepare_context_data(entry_idx, context_h)
                        algo_results, exec_time = self.run_algorithm(context_data, 100)
                        times.append(exec_time)
                        
                        signal, consensus = self.calculate_consensus(algo_results)
                        
                        if signal in ["LONG", "SHORT"] and consensus >= 0.80:
                            trade = self.simulate_trade(entry_idx, signal, 0.80)
                            trade['consensus'] = consensus
                            trade['context_hours'] = context_h
                            trades.append(trade)
            
            # Calculate metrics
            avg_time = np.mean([t for t in times if t < 999]) if times else 999.0
            
            if trades:
                df = pd.DataFrame(trades)
                win_rate = len(df[df['return_pct'] > 0]) / len(df)
                avg_return = df['return_pct'].mean()
                
                results.append({
                    'context_hours': context_h,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'avg_time': avg_time
                })
                
                print(f"  {len(trades)} trades, {win_rate:.1%} win rate, {avg_return:.2f}% avg return, {avg_time:.1f}s")
            else:
                results.append({
                    'context_hours': context_h,
                    'trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'avg_time': avg_time
                })
                print(f"  No trades, {avg_time:.1f}s avg time")
        
        return results
    
    def test_strategy_viability(self, num_tests: int = 50):
        """Test overall strategy viability"""
        print(f"\n=== STRATEGY VIABILITY TEST ===")
        
        test_days = random.sample(self.valid_days, min(num_tests, len(self.valid_days)))
        all_trades = []
        
        print(f"Testing strategy on {len(test_days)} random days...")
        
        for i, date in enumerate(test_days):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_days)}")
                
            day_data = self.msft_data[self.msft_data['date'] == date]
            trading_indices = day_data[
                (day_data['time'] >= datetime.strptime("15:00:00", "%H:%M:%S").time()) &
                (day_data['time'] <= datetime.strptime("15:10:00", "%H:%M:%S").time())
            ].index
            
            if len(trading_indices) > 0:
                entry_idx = random.choice(trading_indices)
                
                context_data = self.prepare_context_data(entry_idx, 12)
                algo_results, _ = self.run_algorithm(context_data, 100)
                signal, consensus = self.calculate_consensus(algo_results)
                
                if signal in ["LONG", "SHORT"] and consensus >= 0.80:
                    trade = self.simulate_trade(entry_idx, signal, 0.80)
                    trade['consensus'] = consensus
                    all_trades.append(trade)
        
        # Analyze results
        if all_trades:
            df = pd.DataFrame(all_trades)
            
            total_trades = len(df)
            winning_trades = len(df[df['return_pct'] > 0])
            win_rate = winning_trades / total_trades
            avg_return = df['return_pct'].mean()
            total_return = df['return_pct'].sum()
            best_trade = df['return_pct'].max()
            worst_trade = df['return_pct'].min()
            avg_duration = df['duration_minutes'].mean()
            
            print(f"\nSTRATEGY RESULTS:")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Best Trade: {best_trade:.2f}%")
            print(f"Worst Trade: {worst_trade:.2f}%")
            print(f"Average Duration: {avg_duration:.1f} minutes")
            
            # Save detailed results
            df.to_csv('strategy_viability_results.csv', index=False)
            print(f"Detailed results saved to strategy_viability_results.csv")
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_return': total_return
            }
        else:
            print("No trades executed!")
            return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0, 'total_return': 0}
    
    def debug_pattern_matching(self):
        """Debug what happens after time matching - why do 1035 matches become 0 results?"""
        print(f"\n" + "="*50)
        print("PATTERN MATCHING DEBUG")
        print("="*50)
        
        # Get sample context data that we know has time matches
        sample_date = self.valid_days[0]
        day_data = self.msft_data[self.msft_data['date'] == sample_date]
        trading_indices = day_data[
            (day_data['time'] >= datetime.strptime("15:00:00", "%H:%M:%S").time()) &
            (day_data['time'] <= datetime.strptime("15:30:00", "%H:%M:%S").time())
        ].index
        
        if len(trading_indices) > 0:
            test_idx = trading_indices[5]
            context_data = self.prepare_context_data(test_idx, 12)
            
            print(f"[DEBUG] Context data shape: {context_data.shape}")
            print(f"[DEBUG] Context data sample:")
            print(f"  First row: time={context_data[0,0]}, close_z={context_data[0,1]:.2f}, vol_z={context_data[0,2]:.2f}")
            print(f"  Last row: time={context_data[-1,0]}, close_z={context_data[-1,1]:.2f}, vol_z={context_data[-1,2]:.2f}")
            
            # Check for NaN or infinite values in context data
            close_zscores = [float(x) for x in context_data[:, 1]]
            volume_zscores = [float(x) for x in context_data[:, 2]]
            
            close_nan_count = sum(1 for x in close_zscores if np.isnan(x) or np.isinf(x))
            volume_nan_count = sum(1 for x in volume_zscores if np.isnan(x) or np.isinf(x))
            
            print(f"[DEBUG] Context data quality check:")
            print(f"  Close z-scores with NaN/Inf: {close_nan_count}/{len(close_zscores)}")
            print(f"  Volume z-scores with NaN/Inf: {volume_nan_count}/{len(volume_zscores)}")
            print(f"  Close z-score range: {min(close_zscores):.2f} to {max(close_zscores):.2f}")
            print(f"  Volume z-score range: {min(volume_zscores):.2f} to {max(volume_zscores):.2f}")
            
            if close_nan_count > 0 or volume_nan_count > 0:
                print(f"[ERROR] Found NaN/Inf values in context data - this could break pattern matching!")
                return
            
            # Test a single FRD500 file to see what happens during pattern processing
            frd_files = os.listdir(self.frd500_path)
            test_file = os.path.join(self.frd500_path, frd_files[0])
            
            print(f"\n[DEBUG] Testing pattern matching with file: {frd_files[0]}")
            
            try:
                # Simulate what ALGOdt4.py does
                frd_data = pd.read_csv(test_file, parse_dates=['datetime'])
                frd_data["time"] = frd_data["datetime"].dt.time
                
                target_time = context_data[-1, 0].time()
                context_length = len(context_data)
                
                print(f"[DEBUG] Looking for patterns ending at {target_time} with context length {context_length}")
                
                # Find rows that end at our target time
                matching_rows = []
                for i, row_time in enumerate(frd_data["time"]):
                    if i >= context_length:  # Need enough history
                        if row_time == target_time:
                            matching_rows.append(i)
                
                print(f"[DEBUG] Found {len(matching_rows)} potential pattern matches in this file")
                
                if len(matching_rows) > 0:
                    # Test pattern matching on first few matches
                    test_matches = matching_rows[:3]  # Test first 3
                    
                    for match_idx in test_matches:
                        print(f"\n[DEBUG] Testing pattern match at index {match_idx}")
                        
                        # Extract the pattern data (like ALGOdt4.py does)
                        start_idx = match_idx - context_length + 1
                        pattern_close = frd_data.iloc[start_idx:match_idx + 1]['close'].values
                        pattern_volume = frd_data.iloc[start_idx:match_idx + 1]['volume'].values
                        
                        print(f"[DEBUG] Pattern data extracted: {len(pattern_close)} close values, {len(pattern_volume)} volume values")
                        
                        # Calculate z-scores for pattern (like ALGOdt4.py does)
                        pattern_close_mean = np.mean(pattern_close)
                        pattern_close_std = np.std(pattern_close)
                        pattern_volume_mean = np.mean(pattern_volume)
                        pattern_volume_std = np.std(pattern_volume)
                        
                        if pattern_close_std == 0 or pattern_volume_std == 0:
                            print(f"[WARNING] Pattern has zero std deviation - skipping")
                            continue
                            
                        pattern_close_z = 4 * (pattern_close - pattern_close_mean) / pattern_close_std
                        pattern_volume_z = (pattern_volume - pattern_volume_mean) / pattern_volume_std
                        
                        print(f"[DEBUG] Pattern z-scores calculated:")
                        print(f"  Close z range: {pattern_close_z.min():.2f} to {pattern_close_z.max():.2f}")
                        print(f"  Volume z range: {pattern_volume_z.min():.2f} to {pattern_volume_z.max():.2f}")
                        
                        # Calculate Euclidean distance (like ALGOdt4.py does)
                        # Create weighted z-scores
                        weights = np.exp(np.linspace(1, 2, context_length))
                        
                        # Our context data (weighted)
                        our_close_z = np.array([float(x) for x in context_data[:, 1]]) * weights
                        our_volume_z = np.array([float(x) for x in context_data[:, 2]]) * weights
                        
                        # Pattern data (weighted)  
                        pattern_close_z_weighted = pattern_close_z * weights
                        pattern_volume_z_weighted = pattern_volume_z * weights
                        
                        # Calculate distance
                        close_diff = our_close_z - pattern_close_z_weighted
                        volume_diff = our_volume_z - pattern_volume_z_weighted
                        
                        distance = np.sqrt(np.sum(close_diff**2) + np.sum(volume_diff**2))
                        
                        print(f"[DEBUG] Euclidean distance calculated: {distance:.2f}")
                        
                        if np.isnan(distance) or np.isinf(distance):
                            print(f"[ERROR] Distance is NaN/Inf - this could be the problem!")
                        else:
                            print(f"[SUCCESS] Valid distance calculated")
                            
                        # Check for any issues in the calculation
                        close_diff_nan = np.sum(np.isnan(close_diff))
                        volume_diff_nan = np.sum(np.isnan(volume_diff))
                        
                        if close_diff_nan > 0 or volume_diff_nan > 0:
                            print(f"[ERROR] Found {close_diff_nan + volume_diff_nan} NaN values in difference calculation")
                
            except Exception as e:
                print(f"[ERROR] Pattern matching test failed: {e}")
                import traceback
                traceback.print_exc()
                
    def run_all_tests(self):
        """Run all optimization tests"""
        print("Starting comprehensive strategy testing...")
        
        # Load data
        self.load_data()
        
        if len(self.valid_days) == 0:
            print("ERROR: No valid trading days found!")
            return
        
        # Debug the pattern matching logic
        self.debug_pattern_matching()
        
        print(f"\n[DEBUG] Pattern matching debug completed!")
        print(f"[NEXT STEP] Check if distances are being calculated correctly")

def main():
    print("="*60)
    print("STRATEGY TESTER STARTING")
    print("="*60)
    
    # File paths
    msft_file = "MSFT_full_1min_adjsplit.csv"
    frd500_path = "C:/Users/simon/OneDrive/Documents/BuyIn/Resources/FRD500"
    
    print(f"[DEBUG] MSFT file: {msft_file}")
    print(f"[DEBUG] FRD500 path: {frd500_path}")
    
    # Check files exist
    if not os.path.exists(msft_file):
        print(f"[ERROR] MSFT file not found: {msft_file}")
        print(f"[DEBUG] Current directory: {os.getcwd()}")
        print(f"[DEBUG] Files in current directory: {os.listdir('.')}")
        return
        
    if not os.path.exists(frd500_path):
        print(f"[ERROR] FRD500 path not found: {frd500_path}")
        return
    
    print(f"[SUCCESS] All files found")
    
    try:
        # Run tests
        print(f"[DEBUG] Creating StrategyTester instance...")
        tester = StrategyTester(msft_file, frd500_path)
        
        print(f"[DEBUG] Starting tests...")
        tester.run_all_tests()
        
        print(f"[SUCCESS] All tests completed!")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Main function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()