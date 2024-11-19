# region imports
from AlgorithmImports import *
from scipy import stats
# endregion

#next steps: 
#trade only options with large implied volatility
#optimize butterfly wing spread 
#find good exit time

class ODTE_Options_Research(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2023, 8, 1)
        self.set_cash(100000)
        #The following algorithm selects 0DTE Option contracts for the SPY that fall within X strikes of the underlying price.
        option = self.add_option('SPY', Resolution.HOUR)
        self._symbol = option.Symbol
        option.set_filter(self._filter)

        #iv and hv calculations:
        self.spy = self.add_equity("SPY", Resolution.HOUR).Symbol
        #6 trading hours in a day, so X days => 6X hours
        self.lookback_period = 30
        self.iv_threshold_multiplier = 1.2  # IV must be higher than HV to trade


    def _filter(self, universe):
        return universe.include_weeklys().expiration(0, 0)#.iv(0, 20)


    def t_distribution_spread(self, degrees_of_freedom, mean, scale, percent):
        # Calculate the tail probability for the specified central percentage
        alpha = (1 - percent / 100) / 2
        spread = (stats.t.ppf(1 - alpha, degrees_of_freedom) - stats.t.ppf(alpha, degrees_of_freedom)) * scale
        return spread

    
    def calculate_historical_volatility(self):
        history = self.history(self.spy, self.lookback_period, Resolution.DAILY)
        if len(history) < self.lookback_period:
            return None
        daily_returns = history['close'].pct_change().dropna()
        # Calculate annualized historical volatility (standard deviation * sqrt(252 trading days))
        hv = daily_returns.std() * (252 ** 0.5)
        return hv

    def set_wing_spread(self, time, confidence = 95):
        param_dict = {10: (2.724, 0.0394, 0.5486), 11: (2.6739, 0.0345,0.4739), 12: (2.4785, 0.0339,0.4057), 13: (2.2987, 0.0246, 0.3525), 14: (2.1135, 0.0197, 0.2938), 15: (2.0658, 0.0120, 0.2170), 16: (2.0658, 0.0120, 0.2170) }
        #assume time is 10:00am
        degrees_of_freedom, mean, scale = param_dict[time]
        return self.t_distribution_spread(degrees_of_freedom, mean, scale, confidence)





    def on_data(self, slice: Slice) -> None:
        if self.portfolio.invested:
            return

        # Get the OptionChain
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return

        
        # Calculate Historical Volatility
        historical_volatility = self.calculate_historical_volatility()
        if historical_volatility is None:
            return  # Skip if not enough data to calculate HV
        # Calculate average implied volatility (IV)
        avg_iv = sum(x.implied_volatility for x in chain) / len([i for i in chain])
        
        # Trade only if IV is significantly larger than annualized HV
        if avg_iv < self.iv_threshold_multiplier * historical_volatility:
            self.debug("Skipping trade as IV is not significantly higher than HV")
            return


        # Select expiry
        expiry = max([x.expiry for x in chain])
        self.debug(f"Expiry: {expiry}")

        # Separate the call and put contracts
        calls = [i for i in chain if i.right == OptionRight.CALL and i.expiry == expiry]
        puts = [i for i in chain if i.right == OptionRight.PUT and i.expiry == expiry]
        if not calls or not puts:
            return

        wing_spread_pct = self.set_wing_spread(self.Time.hour)
        # Calculate ATM strike
        atm_strike = sorted(calls, key=lambda x: abs(chain.underlying.price - x.strike))[0].strike
        wing_spread = wing_spread_pct*atm_strike
        otm_put_strike = atm_strike - wing_spread
        otm_call_strike = atm_strike + wing_spread

        # Verify if the desired strikes exist in the options chain
        available_put_strikes = [x.strike for x in puts]
        available_call_strikes = [x.strike for x in calls]

        # Check for exact match, otherwise fall back
        if otm_put_strike not in available_put_strikes or otm_call_strike not in available_call_strikes:
            # Fallback to minimum OTM put and calculated OTM call strikes
            otm_put_strike = min(available_put_strikes)
            otm_call_strike = 2 * atm_strike - otm_put_strike
            self.debug(f"Fallback to minimum OTM: ATM Strike: {atm_strike}, OTM Put: {otm_put_strike}, OTM Call: {otm_call_strike}")
        else:
            self.debug(f"Using wing spread: ATM Strike: {atm_strike}, OTM Put: {otm_put_strike}, OTM Call: {otm_call_strike}")

        # Check again if fallback strikes are in the available contracts
        if otm_put_strike in available_put_strikes and otm_call_strike in available_call_strikes:
            # Create and execute the iron butterfly strategy
            iron_butterfly = OptionStrategies.iron_butterfly(self._symbol, otm_put_strike, atm_strike, otm_call_strike, expiry)
            self.buy(iron_butterfly, 20)
        else:
            self.debug("Suitable strikes not found for the iron butterfly; skipping trade")
    

    #WORK TO BE DONE
    #1) find way to make bets % of portfolio
    #   a) search documentaion
    #   b) need function to determine percentage of capital to bet:
            # a) need way to calculate expected win rate
            # b) need way to calculate expected profit and loss

    #2) need better threshold calculation of diff between IV ad HV
    #   a) look at historical data about difference between IV and HV, if within 50% percentile, then invest
