
#next steps: 
# if transaction cost = loss => close?
# if volatility go up, then long iron butterfly
# if volatitlity go down, then short
#find good exit time

#do research into historical diff between implied and realized vol: build different thresholds for different market conditions
#look at event buffers

    #WORK TO BE DONE
    #   a) search documentaion

    #FIND WAY TO CALCULATE LIQUIDSTED PFORIT
    #and handle margin calls

from AlgorithmImports import *
from scipy import stats
from collections import deque
import numpy as np
from scipy.stats import t

class ODTE_Options_Research(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(1000000)

        # Add SPY option data with hourly resolution
        option = self.AddOption('SPY', Resolution.Hour)
        self._symbol = option.Symbol
        option.SetFilter(self._filter)

        # Running window for price data
        self.lookback_period = 30  # Lookback period for historical volatility
        self.price_window = deque(maxlen=self.lookback_period)

        # Running window for IV/HV ratios
        self.window_len = 200
        self.running_window = deque(maxlen=self.window_len)

        # Warm-up period for sufficient data collection
        self.SetWarmUp(self.window_len, Resolution.Hour)

        # Add SPY equity data
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        
        # Parameters
        self.kelly_norm_factor = 1
        self.stop_loss_ratio = 1/5  # Adjust as needed
        self.param_dict = {
            10: (1.9241292728, 0.0001579174, 0.0049384812),
            11: (2.6729722498, 0.0003478702, 0.0047362132),
            12: (1.8960267740, 0.0002372966, 0.0037117045),
            13: (2.2975703292, 0.0002469925, 0.0035227054),
            14: (1.9214315186, 0.0001726382, 0.0028405084),
            15: (1.9448810028, 0.0001112350, 0.0021241173)
        }

        self.position = None  # Initialize with no active position

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

    def select_best_iron_butterfly(self, chain, atm_strike):
        """
        Constructs multiple iron butterfly positions with varying wing spreads,
        calculates the expected value for each, and returns the one with the highest EV.
        """
        expiry = max([x.Expiry for x in chain])
        # Get available strikes
        available_strikes = sorted(set([contract.Strike for contract in chain]))
        # Determine possible symmetric wing spreads
        distances = [strike - atm_strike for strike in available_strikes]
        positive_distances = sorted(set([d for d in distances if d > 0]))
        possible_wing_spreads = [d for d in positive_distances if -d in distances]
        p_deg_freedom, p_mean, p_scale = self.create_t_distribution(atm_strike)


        if not possible_wing_spreads:
            self.Debug("No symmetric wing spreads available.")
            return None, None

        num_spreads = min(5, len(possible_wing_spreads))
        if num_spreads < 1:
            self.Debug("Not enough wing spreads available.")
            return None, None
        # Evenly select wing spreads from the possible ones
        indices = np.linspace(0, len(possible_wing_spreads) - 1, num_spreads).astype(int)
        wing_spreads = [possible_wing_spreads[i] for i in indices]
        #self.Debug(f"possible wing spreads: {wing_spreads}")
        best_ev = float('-inf')
        best_position = None

        for wing_spread in wing_spreads:
            target_put_strike = atm_strike - wing_spread
            target_call_strike = atm_strike + wing_spread
            # Find the closest available strikes to the target strikes
            put_strikes = [s for s in available_strikes if s <= target_put_strike]
            call_strikes = [s for s in available_strikes if s >= target_call_strike]
            if not put_strikes or not call_strikes:
                continue
            closest_put = max(put_strikes)
            closest_call = min(call_strikes)
            # Ensure symmetry
            if abs(closest_put - atm_strike) != abs(closest_call - atm_strike):
                continue
            iron_butterfly = OptionStrategies.IronButterfly(
                self._symbol, closest_put, atm_strike, closest_call, expiry
            )
            position = self.compute_iron_butterfly_metrics(iron_butterfly, chain, atm_strike)
            if not position:
                continue
            curr_ev = self.calculate_EV(p_deg_freedom, p_mean, p_scale, position.payoff)
            # Print premiums and strikes for debugging
            # self.Debug(f"Wing spread {wing_spread}, EV is {curr_ev}")
            # self.Debug(f"selected {closest_put} and {closest_call} with atm being {atm_strike}, diff is {atm_strike - closest_put}")
            # #check payoff func
            # self.Debug(f"net premium: {position.net_premium_received}, max_loss is: {position.max_loss}")
            if curr_ev > best_ev:
                best_ev = curr_ev
                best_position = position

        if best_position and best_ev > 0:
            return best_position, best_ev
        else:
            #self.Debug("No suitable iron butterfly position found.")
            return None, None

    class Position:
        def __init__(self, iron_butterfly, chain, net_premium_received, max_loss, m, fee,
                     C_0_ATM, P_0_ATM, C_0_OTM, P_0_OTM, K_C_ATM, K_P_ATM, K_C_OTM, K_P_OTM,
                     atm_strike, entry_time, stop_loss_ratio):
            self.iron_butterfly = iron_butterfly
            self.chain = chain
            self.net_premium_received = net_premium_received
            self.max_loss = max_loss # is positive, needs to mult*-1 be be actual payoff val
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
            self.stop_loss_ratio = stop_loss_ratio
            self.stop_loss = self.stop_loss_ratio * self.max_loss # is positive, needs to mult*-1 be be actual payoff val
            # Calculate price bounds for stop loss
            self.lower_bound, self.upper_bound = self.calculate_price_bounds(-self.stop_loss)
            self.lower_break_even, self.upper_break_even = self.calculate_price_bounds(0)

        def payoff(self, S_T):
            # Option values at expiration
            C_T_OTM = np.maximum(S_T - self.K_C_OTM, 0)
            C_T_ATM = np.maximum(S_T - self.K_C_ATM, 0)
            P_T_OTM = np.maximum(self.K_P_OTM - S_T, 0)
            P_T_ATM = np.maximum(self.K_P_ATM - S_T, 0)

            # Original payoff calculation
            P_T = ((C_T_OTM + P_T_OTM - C_T_ATM - P_T_ATM -
                    (self.C_0_OTM + self.P_0_OTM - self.C_0_ATM - self.P_0_ATM)) * self.m) - self.fee
            # Adjust payoff for stop loss
            P_T = np.maximum(P_T, -self.stop_loss) # Negative because loss is negative payoff

            return P_T

        def calculate_price_bounds(self, payoff_value):
            """
            Calculate the lower and upper bounds where the payoff equals the given value (e.g., stop-loss or break-even).
            """
            x1 = self.K_C_ATM
            x2 = self.K_P_OTM
            y1 = self.net_premium_received - self.fee
            y2 = -self.max_loss
            y3 = payoff_value
            if x2 == x1:
                raise ValueError("x1 and x2 cannot be the same; vertical line slope is undefined.")
            m = (y2 - y1) / (x2 - x1)
            # Calculate x3 based on y3
            x3 = (y3 - y1) / m + x1
            return x3, 2*x1-x3

    def compute_iron_butterfly_metrics(self, iron_butterfly, chain, atm_strike):
        m = 100  # Contract multiplier for options
        fee = 8  # Set fees according to your brokerage model
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
        max_loss = ((wing_width - net_premium_received_per_contract) * m) + fee #because max_loss is +, to turn it into payoff need to multiply by -1
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
            entry_time=self.Time,
            stop_loss_ratio=self.stop_loss_ratio
        )
        return position

    def probability_between_bounds(self, lb, ub, atm):
        new_degrees_of_freedom, new_mean, new_scale = self.create_t_distribution(atm)
        distribution = t(df=new_degrees_of_freedom, loc=new_mean, scale=new_scale)
        # Calculate the probability between the bounds
        return distribution.cdf(ub) - distribution.cdf(lb)

    def create_t_distribution(self, atm):
        """
        Constructs the t-distribution for the final asset price given the % change distribution.
        """
        degrees_of_freedom, mean, scale = self.param_dict[self.Time.hour]

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
        lb = 0.001
        ub = 1 - lb
        x = np.linspace(distribution.ppf(lb), distribution.ppf(ub), 5000)
        #self.Debug(f"Calculating EV from {distribution.ppf(lb)} to {distribution.ppf(ub)}")
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
            atm_strike = sorted(chain, key=lambda x: abs(x.Strike - chain.Underlying.Price))[0].Strike
            avg_iv = sum(x.ImpliedVolatility for x in chain) / len([i for i in chain])
            if historical_volatility and historical_volatility > 0:
                iv_hv_ratio = avg_iv / historical_volatility
                self.running_window.append(iv_hv_ratio)
        else:
            return

        if self.IsWarmingUp or self.Time.hour == 16:
            return

        if self.Portfolio.Invested:
            if self.position is not None:
                # Implement stop loss check
                current_price = self.Securities[self.spy].Price
                if current_price <= self.position.lower_bound or current_price >= self.position.upper_bound:
                    self.Debug(f"Stop loss hit at price {current_price}. Closing position entered at {self.position.entry_time}")
                    self.Liquidate()
                    self.Debug(f"Total portfolio value after liquidating: {self.Portfolio.TotalPortfolioValue}")
                    self.position = None
                    return  # Exit after liquidation
                # Recalculate EV with adjusted payoff function. This is because payoff function for position stays same but price changes every hour:
                # Thus, we have to recalculate EV every hour with new prob dist of price and old payoff
                p_deg_freedom, p_mean, p_scale = self.create_t_distribution(atm_strike)
                curr_EV = self.calculate_EV(p_deg_freedom, p_mean, p_scale, self.position.payoff)
                if curr_EV < 0:
                    self.Debug(f"Closing position entered at {self.position.entry_time} due to low EV")
                    self.Liquidate()
                    self.Debug(f"Total portfolio value after liquidating:{self.Portfolio.TotalPortfolioValue}")
                    self.position = None
            return  # Since we are invested, do not enter new positions

        iv_hv_ratio = avg_iv / historical_volatility
        iv_hv_lower_bound = 50
        percentile = np.percentile(self.running_window, iv_hv_lower_bound)
        if iv_hv_ratio < percentile:
            return
        
        position, EV = self.select_best_iron_butterfly(chain, atm_strike)
        if position is None:
            return
        win_chance = self.probability_between_bounds(position.lower_break_even, position.upper_break_even, atm_strike)
        if win_chance < 0.7 or EV < 50:
            return

        # Adjusted max_loss for Kelly criterion
        adjusted_max_loss = position.stop_loss #this is positive
        loss_amount = adjusted_max_loss
        win_amount = (EV + loss_amount*(1-win_chance))/win_chance
        # Calculate Kelly Criterion Percent with adjusted max_loss
        capital_risked_pct = ((win_chance*win_amount) - ((1-win_chance)*(loss_amount)))/(win_amount*loss_amount)
        # Regularize and cap at 15% for risk management, and prevent % from being zero
        capital_risked_pct = max(0, min(capital_risked_pct * self.kelly_norm_factor, 0.10))
        no_trades = min(100, int((capital_risked_pct * self.Portfolio.TotalPortfolioValue) / (position.stop_loss)))

        self.Debug(f"pct of winning {100*win_chance}%, EV is {EV}, capital_risked_pct is {capital_risked_pct} with trades {no_trades}")

        if no_trades == 0:
            return
        #self.Debug(f"Wing spread: ATM Strike: {atm_strike}, OTM Put: {position.K_P_OTM}, OTM Call: {position.K_C_OTM}, IV/HV Ratio: {iv_hv_ratio:.2f}")
        #self.Debug(f"Max Loss reached at otm put and call: {position.payoff(position.K_P_OTM)}, {position.payoff(position.K_C_OTM)}")
        #self.Debug(f"Stop Loss bounds are: {position.lower_bound}, {position.upper_bound}, with val: {position.payoff(position.upper_bound)}")
        #self.Debug(f"Break-even bounds are: {position.lower_break_even}, {position.upper_break_even}, with val: {int(position.payoff(position.lower_break_even))}, win chance {win_chance}")
        self.Debug(f"Premium: {position.net_premium_received}, Stop Loss: {-position.stop_loss}")
        #self.Debug(f"EV trade: {EV}; Risking cpt_pct: {capital_risked_pct*100} %, at {no_trades} trades.")
        self.Debug(f"Trade placed at {self.time}")
        self.Debug(f"Total Portfolio value prior to entering: {self.Portfolio.TotalPortfolioValue}")
        self.Buy(position.iron_butterfly, no_trades)
        self.position = position  # Keep track of the active position

    def OnEndOfDay(self, symbol: Symbol) -> None:
        if self.position is not None:
            self.Liquidate()
            self.position = None
            self.Debug(f"Total portfolio value after liquidating at end of day: {self.Portfolio.TotalPortfolioValue}")
