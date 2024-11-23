#next steps: 
# if transaction cost = loss => close?
# if volatility go up, then long iron butterfly
# if volatitlity go down, then short
#find good exit time

#do research into historical diff between implied and realized vol: build different thresholds for different market conditions
#look at event buffers

# look at transaction costs - reduce frequency of rebalancing

    #WORK TO BE DONE
    #1) find way to make bets % of portfolio
    #   a) search documentaion
    #   b) need function to determine percentage of capital to bet:
            # a) need way to calculate expected win rate
            # b) need way to calculate expected profit and loss



from AlgorithmImports import *
from scipy import stats
from collections import deque
import numpy as np
from scipy.stats import t

class ODTE_Options_Research(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 6, 1)
        self.SetCash(1000000)

        # Add SPY option data with hourly resolution
        option = self.AddOption('SPY', Resolution.Hour)
        self._symbol = option.Symbol
        option.SetFilter(self._filter)

        # Running window for price data
        self.lookback_period = 30  # Lookback period for historical volatility
        self.price_window = deque(maxlen=self.lookback_period)

        # Running window for IV/HV ratios
        self.window_len = 300
        self.running_window = deque(maxlen=self.window_len)

        # Warm-up period for sufficient data collection
        self.SetWarmUp(self.window_len, Resolution.Hour)

        # Add SPY equity data
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        
        # Parameters
        self.confidence = 0.97
        self.kelly_norm_factor = 1/10
        self.stop_loss_ratio = 1/10
        self.param_dict = {
            10: (2.724, 0.0394, 0.5486),
            11: (2.6739, 0.0345, 0.4739),
            12: (2.4785, 0.0339, 0.4057),
            13: (2.2987, 0.0246, 0.3525),
            14: (2.1135, 0.0197, 0.2938),
            15: (2.0658, 0.0120, 0.2170),
        }

        self.positions = []

    def _filter(self, universe):
        """
        Filters option contracts to include only weekly expirations with 0 days to expiry.
        """
        return universe.IncludeWeeklys().Expiration(0, 0)

    def t_distribution_spread(self, degrees_of_freedom, mean, scale, percent):
        """
        Calculates the spread based on a t-distribution for a given confidence level.
        """
        alpha = (1 - percent / 100) / 2
        spread = (stats.t.ppf(1 - alpha, degrees_of_freedom) - stats.t.ppf(alpha, degrees_of_freedom)) * scale
        return spread

    def update_price_window(self, close_price):
        """
        Updates the rolling price window with the latest closing price.
        """
        self.price_window.append(close_price)

    def calculate_historical_volatility(self):
        """
        Calculates annualized historical volatility from the rolling price window.
        Returns None if there is insufficient data.
        """
        if len(self.price_window) < self.lookback_period:
            return None
        prices = np.array(list(self.price_window))
        returns = np.diff(prices) / prices[:-1]
        hv = np.std(returns) * (252 ** 0.5)  # Annualized volatility
        return hv

    def set_wing_spread(self, time):
        """
        Determines the wing spread based on predefined parameters for different times.
        """
        degrees_of_freedom, mean, scale = self.param_dict[time]
        return self.t_distribution_spread(degrees_of_freedom, mean, scale, 100 * self.confidence)

    def select_wing_spreads(self, chain, atm_strike, wing_spread_pct, epsilon=0.5):
        """
        Selects symmetric strike prices for an iron butterfly strategy.
        """
        calls = [i for i in chain if i.Right == OptionRight.Call]
        puts = [i for i in chain if i.Right == OptionRight.Put]
        if not calls or not puts:
            self.Debug("No calls or puts found.")
            return None, None

        available_put_strikes = [x.Strike for x in puts]
        available_call_strikes = [x.Strike for x in calls]

        wing_spread = wing_spread_pct * atm_strike
        valid_put_strikes = [
            strike for strike in available_put_strikes
            if abs(strike - (atm_strike - wing_spread)) <= epsilon
        ]
        valid_call_strikes = [
            strike for strike in available_call_strikes
            if abs(strike - (atm_strike + wing_spread)) <= epsilon
        ]
        symmetric_strikes = [
            (put, call) for put in valid_put_strikes for call in valid_call_strikes
            if abs(put - atm_strike) == abs(call - atm_strike)
        ]
        if not symmetric_strikes:
            self.Debug("No symmetric strikes found within the epsilon range.")
            return None, None
        closest_symmetric_pair = min(
            symmetric_strikes,
            key=lambda pair: abs(pair[0] - (atm_strike - wing_spread))
        )
        return closest_symmetric_pair

    class Position:
        def __init__(self, iron_butterfly, chain, net_premium_received, max_loss, m, fee,
                     C_0_ATM, P_0_ATM, C_0_OTM, P_0_OTM, K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM,
                     atm_strike, entry_time):
            self.iron_butterfly = iron_butterfly
            self.chain = chain
            self.net_premium_received = net_premium_received
            self.max_loss = max_loss
            self.m = m
            self.fee = fee
            self.C_0_ATM = C_0_ATM
            self.P_0_ATM = P_0_ATM
            self.C_0_OTM = C_0_OTM
            self.P_0_OTM = P_0_OTM
            self.K_C_ATM = K_C_ATM
            self.K_P_ATM = K_P_ATM
            self.K_C_OTM = K_C_OTM
            self.K_P_OTM = K_P_OTM
            self.atm_strike = atm_strike
            self.entry_time = entry_time
            self.is_active = True

        def payoff(self, S_T):
            # Option values at expiration
            C_T_OTM = np.maximum(S_T - self.K_C_OTM, 0)
            C_T_ATM = np.maximum(S_T - self.K_C_ATM, 0)
            P_T_OTM = np.maximum(self.K_P_OTM - S_T, 0)
            P_T_ATM = np.maximum(self.K_P_ATM - S_T, 0)

            # Payoff calculation
            P_T = ((C_T_OTM + P_T_OTM - C_T_ATM - P_T_ATM -
                    (self.C_0_OTM + self.P_0_OTM - self.C_0_ATM - self.P_0_ATM)) * self.m) - self.fee
            return P_T

    def compute_iron_butterfly_metrics(self, iron_butterfly, chain, atm_strike):
        m = 100  # Contract multiplier for options
        fee = 0.4  # Set fees according to your brokerage model
        # Initialize variables to store option premiums and strikes
        C_0_ATM = P_0_ATM = C_0_OTM = P_0_OTM = None
        K_C_ATM = K_P_ATM = K_C_OTM = K_P_OTM = None
        # Identify ATM and OTM strikes from the legs
        for leg in iron_butterfly.OptionLegs:
            # Find the contract in the chain
            contract = next(
                (x for x in chain if x.Strike == leg.Strike and x.Right == leg.Right and x.Expiry.date() == leg.Expiration.date()),
                None
            )
            if contract is None:
                self.Debug(f"Contract not found for leg: Strike {leg.Strike}, Right {leg.Right}, Expiration {leg.Expiration}")
                continue
            # Get option price (mid-price between bid and ask)
            option_price = (contract.BidPrice + contract.AskPrice) / 2
            if option_price <= 0:
                self.Debug(f"Option price is zero or negative for {contract.Symbol}")
                continue
            # Map premiums and strikes based on leg type
            if leg.Quantity == -1 and leg.Right == OptionRight.Call:
                C_0_ATM = option_price
                K_C_ATM = leg.Strike
            elif leg.Quantity == -1 and leg.Right == OptionRight.Put:
                P_0_ATM = option_price
                K_P_ATM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Call:
                C_0_OTM = option_price
                K_C_OTM = leg.Strike
            elif leg.Quantity == 1 and leg.Right == OptionRight.Put:
                P_0_OTM = option_price
                K_P_OTM = leg.Strike
        # Check if all necessary premiums and strikes are collected
        if None in [C_0_ATM, P_0_ATM, C_0_OTM, P_0_OTM, K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM]:
            self.Debug("Incomplete data to compute metrics.")
            return None
        # Net premium received per contract (without multiplier)
        net_premium_received_per_contract = (C_0_ATM + P_0_ATM) - (C_0_OTM + P_0_OTM)
        net_premium_received = net_premium_received_per_contract * m
        max_profit = net_premium_received - fee
        # Wing width (difference between strikes)
        wing_width = K_C_OTM - K_C_ATM
        # Maximum loss
        max_loss = ((wing_width - net_premium_received_per_contract) * m) - fee
        # Create Position object
        position = self.Position(
            iron_butterfly=iron_butterfly,
            chain=chain,
            net_premium_received=net_premium_received,
            max_loss=max_loss,
            m=m,
            fee=fee,
            C_0_ATM=C_0_ATM,
            P_0_ATM=P_0_ATM,
            C_0_OTM=C_0_OTM,
            P_0_OTM=P_0_OTM,
            K_C_ATM=K_C_ATM,
            K_P_ATM=K_P_ATM,
            K_C_OTM=K_C_OTM,
            K_P_OTM=K_P_OTM,
            atm_strike=atm_strike,
            entry_time=self.Time
        )
        return position

    def create_t_distribution(self, degrees_of_freedom, mean, scale, atm):
        """
        Constructs the t-distribution for the final asset price given the % change distribution.
        """
        new_degrees_of_freedom = degrees_of_freedom  # Degrees of freedom remain the same
        new_mean = atm * (1 + mean)  # Adjust mean for final price
        new_scale = atm * scale  # Adjust scale for final price

        return new_degrees_of_freedom, new_mean, new_scale

    def calculate_EV(self, new_degrees_of_freedom, new_mean, new_scale, payoff):
        """
        Calculates the expected value of the payoff function under the given distribution.
        """
        # Define the Student's t-distribution for the final price
        distribution = t(df=new_degrees_of_freedom, loc=new_mean, scale=new_scale)
        
        # Approximate expected value using numerical integration
        x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 10000)
        pdf = distribution.pdf(x)
        EV = np.sum(payoff(x) * pdf * (x[1] - x[0]))

        return EV

    def OnData(self, slice: Slice) -> None:
        spy_data = slice.Bars.get(self.spy)
        if spy_data:
            self.update_price_window(spy_data.Close)

        historical_volatility = self.calculate_historical_volatility()
        chain = slice.OptionChains.get(self._symbol, None)
        if chain:
            avg_iv = sum(x.ImpliedVolatility for x in chain) / len([i for i in chain])
            if historical_volatility and historical_volatility > 0:
                iv_hv_ratio = avg_iv / historical_volatility
                self.running_window.append(iv_hv_ratio)
        else:
            return

        if self.IsWarmingUp or self.Time.hour == 16:
            return

        if self.Portfolio.Invested:
            for position in self.positions:
                if position.is_active:
                    degrees_of_freedom, mean, scale = self.param_dict.get(self.Time.hour, (2.0658, 0.0120, 0.2170))
                    p_deg_freedom, p_mean, p_scale = self.create_t_distribution(degrees_of_freedom, mean, scale, position.atm_strike)
                    curr_EV = self.calculate_EV(p_deg_freedom, p_mean, p_scale, position.payoff)
                    if curr_EV < 30:
                        self.Debug(f"Closing position entered at {position.entry_time}")
                        self.Debug(f"Total portfolio cash before liquidating: {self.Portfolio.Cash}")
                        self.Liquidate(position.iron_butterfly)
                        self.Debug(f"Total portfolio cash after liquidating: {self.Portfolio.Cash}")
                        position.is_active = False
            return

        iv_hv_ratio = avg_iv / historical_volatility
        iv_hv_lower_bound = 50
        percentile = np.percentile(self.running_window, iv_hv_lower_bound)
        if iv_hv_ratio < percentile:
            self.Debug(f"Skipping trade: IV/HV ratio ({iv_hv_ratio:.2f}) below {iv_hv_lower_bound}th percentile ({percentile:.2f})")
            return

        expiry = max([x.Expiry for x in chain])
        atm_strike = sorted(chain, key=lambda x: abs(x.Strike - chain.Underlying.Price))[0].Strike
        wing_spread_pct = self.set_wing_spread(self.Time.hour) / 100
        closest_otm_put_strike, closest_otm_call_strike = self.select_wing_spreads(chain, atm_strike, wing_spread_pct)

        if closest_otm_put_strike is None or closest_otm_call_strike is None:
            return
        
        iron_butterfly = OptionStrategies.IronButterfly(
            self._symbol, closest_otm_put_strike, atm_strike, closest_otm_call_strike, expiry
        )
        self.Debug(
            f"Placing Iron Butterfly trade at {self.Time.date()}.\n"
            f"Using wing spread within epsilon: ATM Strike: {atm_strike}, OTM Put: {closest_otm_put_strike}, OTM Call: {closest_otm_call_strike}, IV/HV Ratio: {iv_hv_ratio:.2f}"
        )

        position = self.compute_iron_butterfly_metrics(iron_butterfly, chain, atm_strike)
        if position is None:
            return

        degrees_of_freedom, mean, scale = self.param_dict[self.Time.hour]
        p_deg_freedom, p_mean, p_scale = self.create_t_distribution(degrees_of_freedom, mean, scale, position.atm_strike)
        curr_EV = self.calculate_EV(p_deg_freedom, p_mean, p_scale, position.payoff)
        if curr_EV < 0:
            self.Debug(f"EV is negative at {self.Time} with value of {curr_EV}, no trade placed")
            return
        self.debug(f"currEV is {curr_EV}, we are going to buy now")
        loss = position.max_loss
        win = (curr_EV - loss * (1 - self.confidence)) / self.confidence
        # Calculate Kelly Criterion Percent
        capital_risked_pct = (self.confidence * loss - (1 - self.confidence) * win) / (win * loss)
        # Regularize and cap at 15% for risk management, and prevent % from being zero
        capital_risked_pct = max(0, min(capital_risked_pct * self.kelly_norm_factor, 0.15))
        self.Debug(f"Capital risked percentage is {capital_risked_pct}")
        no_trades = int((capital_risked_pct * self.Portfolio.TotalPortfolioValue) / loss)

        self.Debug(f"Total portfolio cash before buying: {self.Portfolio.Cash}")
        self.Buy(position.iron_butterfly, no_trades)
        self.Debug(f"Total portfolio cash after buying: {self.Portfolio.Cash}, number of trades placed: {no_trades}, premium: {position.net_premium_received}")

        self.positions.append(position)

    def OnEndOfDay(self, symbol: Symbol) -> None:
        # self.Debug(f"End of day for {symbol} on {self.Time}")
        # self.Debug(f"Total portfolio cash before liquidating at end of day: {self.Portfolio.Cash}")
        # for position in self.positions:
        #     if position.is_active:
        #         self.Liquidate(position.iron_butterfly)
        #         position.is_active = False
        # self.Debug(f"Total portfolio cash after liquidating at end of day: {self.Portfolio.Cash}")
        self.liquidate()
