# region imports
from AlgorithmImports import *
from scipy import stats
from collections import deque
import numpy as np

#next steps: 
# if transaction cost = loss => close?
# if volatility go up, then long iron butterfly
# if volatitlity go down, then short
#find good exit time

#sync up implied volatility and historical volatility
#do research into historical diff between implied and realized vol: build different thresholds for different market conditions
#look at event buffers

# look at transaction costs - reduce frequency of rebalancing

    #WORK TO BE DONE
    #1) find way to make bets % of portfolio
    #   a) search documentaion
    #   b) need function to determine percentage of capital to bet:
            # a) need way to calculate expected win rate
            # b) need way to calculate expected profit and loss

 
class ODTE_Options_Research(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_cash(1000000)
        #The following algorithm selects 0DTE Option contracts for the SPY that fall within X strikes of the underlying price.
        option = self.add_option('SPY', Resolution.HOUR)
        self._symbol = option.Symbol
        option.set_filter(self._filter)
        #create running window
        # Create running window for prices (instead of returns)
        self.lookback_period = 30  # Number of periods for calculating HV
        self.price_window = deque(maxlen=self.lookback_period)

        # Create running window for IV/HV ratios
        self.window_len = 150
        self.running_window = deque(maxlen=self.window_len)
        # Pre-fill the running window
        # Set a warm-up period for historical data collection
        self.SetWarmUp(self.window_len, Resolution.HOUR)  # Approximate enough data

        #iv and hv calculations:
        self.spy = self.add_equity("SPY", Resolution.HOUR).Symbol


    def _filter(self, universe):
        return universe.include_weeklys().expiration(0, 0)


    def t_distribution_spread(self, degrees_of_freedom, mean, scale, percent):
        # Calculate the tail probability for the specified central percentage
        alpha = (1 - percent / 100) / 2
        spread = (stats.t.ppf(1 - alpha, degrees_of_freedom) - stats.t.ppf(alpha, degrees_of_freedom)) * scale
        return spread


    def update_price_window(self, close_price):
        """
        Update the price window with the latest closing price.
        """
        self.price_window.append(close_price)


    def calculate_historical_volatility(self):
        """
        Calculate annualized historical volatility using the price window.
        """
        if len(self.price_window) < self.lookback_period:
            return None
        # Calculate daily returns from the price window
        prices = np.array(list(self.price_window))
        returns = np.diff(prices) / prices[:-1]
        # Calculate standard deviation of returns
        hv = np.std(returns) * (252 ** 0.5)  # Annualized HV
        return hv

    def set_wing_spread(self, time, confidence = 90):
        param_dict = {10: (2.724, 0.0394, 0.5486), 11: (2.6739, 0.0345,0.4739), 12: (2.4785, 0.0339,0.4057), 13: (2.2987, 0.0246, 0.3525), 14: (2.1135, 0.0197, 0.2938), 15: (2.0658, 0.0120, 0.2170), 16: (2.0658, 0.0120, 0.2170) }
        degrees_of_freedom, mean, scale = param_dict[time]
        return self.t_distribution_spread(degrees_of_freedom, mean, scale, confidence)





    def on_data(self, slice: Slice) -> None:
        # Update price window during the warmup period
        spy_data = slice.Bars.get(self.spy)
        if spy_data:
            self.update_price_window(spy_data.Close)

        historical_volatility = self.calculate_historical_volatility()
        chain = slice.option_chains.get(self._symbol, None)
        if chain:
            avg_iv = sum(x.implied_volatility for x in chain) / len([i for i in chain])
            if historical_volatility is not None and historical_volatility > 0:  # Avoid division by zero
                iv_hv_ratio = avg_iv / historical_volatility
                self.running_window.append(iv_hv_ratio)
        else:
            return

        if self.is_warming_up:
            self.debug(f"warmup % is: {len(self.running_window)/self.window_len}")
            return

        if self.portfolio.invested:
            return
        

        iv_hv_ratio = avg_iv / historical_volatility
        # Check if IV/HV ratio is above the 50th percentile of the running window
        percentile = np.percentile(self.running_window, 75)
        if iv_hv_ratio < percentile:
            self.Debug(f"Skipping trade as IV/HV ratio ({iv_hv_ratio:.2f}) is below percentile ({percentile:.2f})")
            return
        self.Debug(f"Trade since IV/HV ratio ({iv_hv_ratio:.2f}) is above percentile ({percentile:.2f})")


        # Select expiry
        expiry = max([x.expiry for x in chain])

        # Separate the call and put contracts
        calls = [i for i in chain if i.right == OptionRight.CALL and i.expiry == expiry]
        puts = [i for i in chain if i.right == OptionRight.PUT and i.expiry == expiry]
        if not calls or not puts:
            return
        available_put_strikes = [x.strike for x in puts]
        available_call_strikes = [x.strike for x in calls]

        wing_spread_pct = self.set_wing_spread(self.Time.hour)/100
        # Calculate ATM strike
        atm_strike = sorted(calls, key=lambda x: abs(chain.underlying.price - x.strike))[0].strike
        self.debug(f"wing spread percent is {wing_spread_pct}")
        wing_spread = wing_spread_pct*atm_strike
        self.debug(f"atm_strike is {atm_strike}, chain.underlying.price is {chain.underlying.price}")
        otm_put_strike = atm_strike - wing_spread
        otm_call_strike = atm_strike + wing_spread

        # Set an epsilon for flexibility in matching strike prices
        epsilon = 0.5  # Adjust epsilon as needed

        # Find strikes within the epsilon range
        valid_put_strikes = [
            strike for strike in available_put_strikes
            if abs(strike - (atm_strike - wing_spread)) <= epsilon
        ]
        valid_call_strikes = [
            strike for strike in available_call_strikes
            if abs(strike - (atm_strike + wing_spread)) <= epsilon
        ]
        # Ensure symmetry: Find pairs of strikes that are equidistant from the ATM strike
        symmetric_strikes = [
            (put, call) for put in valid_put_strikes for call in valid_call_strikes
            if abs(put - atm_strike) == abs(call - atm_strike)
        ]
        if not symmetric_strikes:
            self.Debug("No symmetric strikes found within the epsilon range.")
            return
        # Choose the pair of strikes with the smallest wing spread
        closest_symmetric_pair = min(
            symmetric_strikes,
            key=lambda pair: abs(pair[0] - (atm_strike - wing_spread))
        )
        closest_otm_put_strike, closest_otm_call_strike = closest_symmetric_pair
        # Debugging: Log selected strikes
        self.Debug(f"Selected symmetric strikes: ATM Strike: {atm_strike}, "
                f"OTM Put: {closest_otm_put_strike}, OTM Call: {closest_otm_call_strike}")


        # If no suitable strikes are found, log how far off the closest strikes are
        if closest_otm_put_strike is None or closest_otm_call_strike is None:
            # Find the closest strikes without epsilon constraints
            closest_put_strike_off = min(available_put_strikes, key=lambda x: abs(x - (atm_strike - wing_spread)))
            closest_call_strike_off = min(available_call_strikes, key=lambda x: abs(x - (atm_strike + wing_spread)))
            # Calculate the offsets
            put_offset = abs(closest_put_strike_off - (atm_strike - wing_spread))
            call_offset = abs(closest_call_strike_off - (atm_strike + wing_spread))
            # Log the information
            self.Debug(f"No suitable strikes within epsilon range.")
            self.Debug(f"Closest put strike: {closest_put_strike_off} (target: {(atm_strike - wing_spread):.2f}), "
                    f"Closest call strike: {closest_call_strike_off} (target: {(atm_strike + wing_spread):.2f})")

            return

        # Debugging: Log selected strikes
        self.Debug(f"Using wing spread within epsilon: ATM Strike: {atm_strike}, OTM Put: {closest_otm_put_strike}, OTM Call: {closest_otm_call_strike}")

        # Create and execute the iron butterfly strategy
        iron_butterfly = OptionStrategies.iron_butterfly(self._symbol, closest_otm_put_strike, atm_strike, closest_otm_call_strike, expiry)
        self.debug(f"entering option now {self.time.date()}")
        self.Buy(iron_butterfly, 20)

    def on_end_of_day(self, symbol: Symbol) -> None:
        # Your end-of-day logic here
        self.Debug(f"End of day for {Symbol} on {self.Time}")
        self.liquidate()
